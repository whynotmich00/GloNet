import jax
import jax.numpy as jnp
from jax import random
from tensorflow import data as tf_data
from typing import Optional, List, Any, Dict
from functools import reduce
from jax import tree_util
from statistics import mean


def create_dataloader(x: jnp.ndarray, y: jnp.ndarray, batch_size: int):
    loader = tf_data.Dataset.from_tensor_slices((x, y))
    loader = loader.shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(tf_data.experimental.AUTOTUNE)
    return loader


class MLP:
    def __call__(self, params: dict, x: jnp.ndarray):
        """Apply the MLP to a single input."""
        vmapped_dot = jax.vmap(jnp.dot, (None, 0))
        for i in range(len(params)):
            x = vmapped_dot(params[f"layer{i}"]["w"], x) + params[f"layer{i}"]["b"]
            x = jax.nn.relu(x)
        return x


def OLS(X: jnp.ndarray, y: jnp.ndarray):
    """
    Perform Ordinary Least Squares (OLS) to find the coefficients.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        
    Returns:
        beta: Estimated coefficients of shape (n_features,).
    """
    # Compute (X^T X)
    XT_X = X.T @ X
    
    # Compute (X^T y)
    XT_y = X.T @ y
    
    # Solve for beta = (X^T X)^(-1) X^T y
    beta = jnp.linalg.solve(XT_X, XT_y)
    
    return beta




def simulate_dataset(rng: random.PRNGKey, shape_x: tuple, xmean: float, xvar: float, noise: float):
    x = xmean + jnp.sqrt(xvar) * random.normal(key=rng, shape=shape_x)
    betas = random.normal(key=rng, shape=(x.shape[1]))
    y = jnp.dot(x, betas) + jnp.sqrt(noise)*random.normal(key=rng, shape=(x.shape[0]))
    return x, y[:, None]


def detect_nan(d: dict):
    list_leaves = jax.tree_util.tree_flatten(d)[0]
    return sum(1 if jnp.any(jnp.isnan(i)) else 0 for i in list_leaves)


def mse(params: dict, model, x: jnp.ndarray, y: jnp.ndarray):
    y_hat = model(params=params, x=x) # predict
    assert y.shape == y_hat.shape, "y and y_hat have different shapes"
    return jnp.mean((y - y_hat)**2) # calculate loss


@jax.jit
def mean_gradient(lista_dei_gradienti: List[Any], batch_size: int):
    summed_grads = reduce(lambda a, b: tree_util.tree_map(lambda x, y: x + y, a, b), lista_dei_gradienti)
    return tree_util.tree_map(lambda x: x / batch_size, summed_grads)



def sgd(params: dict, gradients: dict, learning_rate: float, weight_decay: Optional[float]=None, time_step: int=1):
    if weight_decay:
        gradients = tree_util.tree_map(lambda params, gradients: gradients + weight_decay * params, params, gradients)
    
    return tree_util.tree_map(lambda params, gradients: params - (learning_rate / time_step) * gradients, params, gradients)



def weighted_sgd(params: dict, gradients: dict, learning_rate: float, weight_decay: Optional[float]=None, time_step: int=1):
    if weight_decay:
        gradients = tree_util.tree_map(lambda params, gradients: gradients + weight_decay * params, params, gradients)
    
    return tree_util.tree_map(lambda params, gradients: params - (learning_rate / time_step) * gradients, params, gradients)



# @jax.jit
def train_step(params: dict, model, x: jnp.ndarray, y: jnp.ndarray, learning_rate: float=1e-3, weight_decay: float=None, time_step: int=1):
    loss, gradients = jax.value_and_grad(mse)(params, model=model, x=x, y=y)
    
    # Average gradients over the batch
    batch_size = x.shape[0]
    gradients = tree_util.tree_map(lambda g: g / batch_size, gradients)
    
    max_grad = max(jnp.max(i) for i in tree_util.tree_flatten(gradients)[0])
    min_grad = min(jnp.min(i) for i in tree_util.tree_flatten(gradients)[0])
    
    if max_grad is jnp.nan:
        print(f"max grad {max_grad}")
    
    if min_grad is jnp.nan:
        print(f"min grad: {min_grad}")
    
    if detect_nan(gradients):
        print("nan in gradients")
    
    updated_params = sgd(params=params, gradients=gradients, learning_rate=learning_rate, weight_decay=weight_decay, time_step=time_step)
    return loss, updated_params, gradients



def running_mean(x: jnp.ndarray, window: int):
    return [jnp.mean(x[window*(i): window*(i+1)]) for i in jnp.arange(0, x.shape[0])]



def R_squared(params: dict, model, x: jnp.ndarray, y: jnp.ndarray):
    y_hat = model(params, x)
    return jnp.var(y_hat) / jnp.var(y)



class MetricTracker:
    def __init__(self, name: str):
        """
        Generic tracker for metrics such as Loss and Accuracy.
        """
        self.metrics = {"Training": {}, "Validation": {}}
        self.name = name

    def add_entry(self, mode: str, epoch: int, step: int, value):
        """
        Add a metric entry for the given mode, epoch, and step.
        """
        assert mode in ("Training", "Validation"), "Mode must be either 'Training' or 'Validation'."
        
        epoch_key = f"Epoch{epoch}"
        step_key = f"Step{step}"
        
        # Initialize the epoch if it doesn't exist
        if epoch_key not in self.metrics[mode]:
            self.metrics[mode][epoch_key] = {}
        
        # Add the value for the specific step
        self.metrics[mode][epoch_key][step_key] = float(value)

    def mean_on_epochs(self):
        return {
            mode: {
                epoch: mean(self.metrics[mode][epoch].values()) for epoch in self.metrics[mode]
                } 
            for mode in self.metrics
            }

    def __repr__(self):
        return f"{self.name} Tracker: {self.metrics}"


class LossTracker(MetricTracker):
    def __init__(self):
        super().__init__("Loss")

    def __call__(self, mode: str, epoch: int, step: int, loss):
        self.add_entry(mode, epoch, step, loss)


class AccuracyTracker(MetricTracker):
    def __init__(self):
        super().__init__("Accuracy")

    def __call__(self, mode: str, epoch: int, step: int, params: Dict[str, Any], model, images: jnp.ndarray, labels: jnp.ndarray, k: int = 3):
        accuracy = self.top_k_accuracy(params, model, images, labels, k)
        self.add_entry(mode, epoch, step, accuracy)
