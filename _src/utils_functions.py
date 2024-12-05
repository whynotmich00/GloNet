import jax
from jax import random
import jax.numpy as jnp
from flax.training import train_state
import optax
from typing import Any, List
from functools import reduce, partial
from flax import struct
from tqdm import tqdm

# Load MNIST data from JAX datasets (you can use other datasets like CIFAR-10 too)
from tensorflow.keras.datasets import mnist
from tensorflow import newaxis
from tensorflow.data import Dataset, AUTOTUNE

from _src.metrics_tracker import LossTracker, AccuracyTracker


# Define the main training loop
def train_model(task, model, train_ds, test_ds, lr, momentum=None, num_epochs=10, track_metrics=True, track_grad_and_params_norms=False):
    # Rng
    rng = random.PRNGKey(0)
    grad_norm_history =  [] if track_grad_and_params_norms else None
    params_norm_history =  [] if track_grad_and_params_norms else None

    # initialize metrics trackers
    loss_tracker = LossTracker()
    accuracy_tracker = AccuracyTracker()
    
    # Train for a few epochs
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1}")
        
        # Training
        for step, (images, labels) in enumerate(train_ds):
            images, labels = jnp.array(images, dtype=jnp.float32), jnp.array(labels, dtype=jnp.float32)
            
            if (step == 0) and (epoch == 0):
                # Initialize the model and optimmizer
                one_image_shape = images[0][None, ...].shape # MNIST images are 28x28x1 
                state = create_train_state(rng, model, one_image_shape, lr=lr, momentum=momentum)
            
            state, loss, accuracy,grads = train_step(state, images, labels, task)
            
            if step % 5 == 0 and track_grad_and_params_norms:
                # compute gradient and params norm and append it to the history
                grad_norm_history.append(compute_norm(grads))
                params_norm_history.append(compute_norm(state.params))
            
            # logs metrics every 10 steps
            if step % 10 == 0 and track_metrics:
                loss_tracker("Training", epoch=epoch, step=step, loss=loss)
                accuracy_tracker("Training", epoch=epoch, step=step, accuracy=accuracy)
        
        
        # Validation
        for step, (images, labels) in enumerate(test_ds):
            images, labels = jnp.array(images, dtype=jnp.float32), jnp.array(labels, dtype=jnp.float32)
            
            loss, accuracy = eval_step(state, images, labels, task)
            
            if step % 10 == 0 and track_metrics:
                loss_tracker("Validation", epoch=epoch, step=step, loss=loss)
                accuracy_tracker("Validation", epoch=epoch, step=step, accuracy=accuracy)

        if track_metrics:
            epoch_key = f"Epoch{epoch}"
            
            loss_mean_on_epoch = loss_tracker.mean_on_epochs()
            accuracy_mean_on_epoch = accuracy_tracker.mean_on_epochs()
            
            print(f"Training loss: {loss_mean_on_epoch["Training"][epoch_key]:.4f}")
            print(f"Training accuracy: {accuracy_mean_on_epoch["Training"][epoch_key]:.4f}")
            
            print(f"Validation loss: {loss_mean_on_epoch["Validation"][epoch_key]:.4f}")
            print(f"Validation accuracy: {accuracy_mean_on_epoch["Validation"][epoch_key]:.4f}")
    
    state = state.replace(grad_norm_history=grad_norm_history, params_norm_history=params_norm_history)
    
    return state, loss_tracker, accuracy_tracker


def get_number_of_parameters(params):
    total = []
    
    for layer in params["params"].keys():
        for par in params["params"][layer].items():
            total.append((layer, reduce(lambda x, y: x * y, par[1].shape)))
    print(total)

def get_mnsit_dataloaders(batch_size, model_type="CNN"):
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the images to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Add a channel dimension for compatibility with convolutional layers
    x_train = x_train[..., newaxis]
    x_test = x_test[..., newaxis]
    
    if model_type == "MLP":
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    
    train_dataset = Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    test_dataset = Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    
    return train_dataset, test_dataset

# Define the cross-entropy loss function
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(one_hot_labels * jax.nn.log_softmax(logits))

# Define the mse loss function
def mse(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return jnp.mean(jnp.sum(jnp.square(one_hot_labels - jax.nn.log_softmax(logits)), axis=-1))

# Define loss base on task
task_loss = lambda logits, labels, task: cross_entropy_loss(logits, labels) if task == "classification" else mse(logits, labels)

# Compute accuracy
def compute_top_k_accuracy(logits, labels, k=1):
    # Get the indices of the top-k predictions
    top_k_predictions = jnp.argsort(logits, axis=-1)[:, -k:]
    
    # Compare true labels against top-k predictions
    correct = jnp.any(top_k_predictions == labels[:, None], axis=-1)
    
    # Compute mean accuracy
    return jnp.mean(correct)

# Compute norm
@jax.jit
def compute_norm(xs):
    return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(xs)))

# Define a function for creating an optimizer
def create_optimizer(lr, momentum):
    return optax.sgd(learning_rate=lr, momentum=momentum)

# Define the training state class to keep track of parameters and optimizer
class TrainState(train_state.TrainState):
    batch_stats: Any = None
    grad_norm_history: List[float] = struct.field(pytree_node=False, default_factory=list)
    params_norm_history: List[float] = struct.field(pytree_node=False, default_factory=list)

# Define a function to initialize the model and optimizer
def create_train_state(rng, model, input_shape, lr, momentum):
    params = model.init(rng, jnp.ones(input_shape))  # Initialize parameters
    get_number_of_parameters(params)
    
    tx = create_optimizer(lr, momentum)  # Initialize the optimizer
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the training step (forward pass + backward pass)
@partial(jax.jit, static_argnames=['task'])
def train_step(state, images, labels, task):
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        loss = task_loss(logits, labels, task=task)
        # loss = cross_entropy_loss(logits, labels)
        accuracy = compute_top_k_accuracy(logits, labels, k=1)
            
        return loss, accuracy
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, accuracy), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss, accuracy, grads

# Define the evaluation step (no gradient computation)
@partial(jax.jit, static_argnames=['task'])
def eval_step(state, images, labels, task):
    logits = state.apply_fn(state.params, images)
    loss = task_loss(logits, labels, task=task)
    accuracy = compute_top_k_accuracy(logits, labels)
    return loss, accuracy
