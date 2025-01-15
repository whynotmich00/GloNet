import jax
from jax import random
import jax.numpy as jnp
from flax.training import train_state
import optax
from typing import Any
from tqdm import tqdm

from _src.metrics_tracker import LossTracker, AccuracyTracker

# Define the training state class to keep track of parameters and optimizer
class TrainState(train_state.TrainState):
    batch_stats: Any = None


# Define the cross-entropy loss function
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(one_hot_labels * jax.nn.log_softmax(logits))

# Compute accuracy
def compute_top_k_accuracy(logits, labels, k=1):
    # Get the indices of the top-k predictions
    top_k_predictions = jnp.argsort(logits, axis=-1)[:, -k:]
    
    # Compare true labels against top-k predictions
    correct = jnp.any(top_k_predictions == labels[:, None], axis=-1)
    
    # Compute mean accuracy
    return jnp.mean(correct)

# Define a function for creating an optimizer
def create_optimizer(config_optimizer):
    if config_optimizer["optimizer"] == "SGD": 
        return optax.sgd(learning_rate=config_optimizer["lr"], momentum=config_optimizer["momentum"])
    
    elif config_optimizer["optimizer"] == "ADAM": 
        return optax.adam(learning_rate=config_optimizer["lr"])
    
    else:
        raise ValueError(f"Optimizer {config_optimizer["optimizer"]} not implemented")

# Define a function to initialize the model and optimizer
def create_train_state(rng, model, input_shape, config_optimizer):
    params = model.init(rng, jnp.ones(input_shape))  # Initialize parameters
    
    tx = create_optimizer(config_optimizer)  # Initialize the optimizer
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the training step (forward pass + backward pass)
@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        loss = cross_entropy_loss(logits, labels)
        accuracy = compute_top_k_accuracy(logits, labels, k=1)
            
        return loss, accuracy
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, accuracy), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss, accuracy

# Define the evaluation step (no gradient computation)
@jax.jit
def eval_step(state, images, labels):
    logits = state.apply_fn(state.params, images)
    loss = cross_entropy_loss(logits, labels)
    accuracy = compute_top_k_accuracy(logits, labels)
    return loss, accuracy


# Define the main training loop
def train_model(model, train_ds, test_ds, config_optimizer, flags):
    # Rng
    rng = random.PRNGKey(0)

    # initialize metrics trackers
    loss_tracker = LossTracker()
    accuracy_tracker = AccuracyTracker()
    
    # Train for a few epochs
    for epoch in tqdm(range(flags.epochs)):
        print(f"Epoch {epoch + 1}")
        
        # Training
        for step, (images, labels) in enumerate(train_ds):
            images, labels = jnp.array(images, dtype=jnp.bfloat16), jnp.array(labels, dtype=jnp.bfloat16)
            
            if (step == 0) and (epoch == 0):
                # Initialize the model and optimmizer
                one_image_shape = images[0][None, ...].shape # MNIST images are 28x28x1 
                state = create_train_state(rng, model, one_image_shape, config_optimizer)
            
            state, loss, accuracy = train_step(state, images, labels)
            
            # logs metrics every 10 steps
            if step % 10 == 0 and flags.track_metrics:
                loss_tracker("Training", epoch=epoch, step=step, loss=loss)
                accuracy_tracker("Training", epoch=epoch, step=step, accuracy=accuracy)
        
        
        # Validation
        for step, (images, labels) in enumerate(test_ds):
            images, labels = jnp.array(images, dtype=jnp.bfloat16), jnp.array(labels, dtype=jnp.bfloat16)
            
            loss, accuracy = eval_step(state, images, labels)
            
            if step % 10 == 0 and flags.track_metrics:
                loss_tracker("Validation", epoch=epoch, step=step, loss=loss)
                accuracy_tracker("Validation", epoch=epoch, step=step, accuracy=accuracy)

        if flags.track_metrics:
            # print the training and validation loss and accuracy means for each epoch
            epoch_key = f"Epoch{epoch}"
            
            loss_mean_on_epoch = loss_tracker.mean_on_epochs()
            accuracy_mean_on_epoch = accuracy_tracker.mean_on_epochs()
            
            print(f"Training loss: {loss_mean_on_epoch["Training"][epoch_key]:.4f}")
            print(f"Training accuracy: {accuracy_mean_on_epoch["Training"][epoch_key]:.4f}")
            
            print(f"Validation loss: {loss_mean_on_epoch["Validation"][epoch_key]:.4f}")
            print(f"Validation accuracy: {accuracy_mean_on_epoch["Validation"][epoch_key]:.4f}")
    
    return state, loss_tracker, accuracy_tracker