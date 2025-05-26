import jax
import jax.numpy as jnp
import jax.random as jrn
import jax.tree_util as jtu
import optax
from tqdm import tqdm
from random import randint
import numpy as np

from _src.get_data import get_mnsit_dataloaders
from _src.utils import create_model, TrainState, create_optimizer

# Imagenette specifics setting
IN_DIM = 784
NUM_CLASSES = 10

# Define a function to initialize the model and optimizer
def create_train_state(flags, model, rng):
    params = model.init(rng, jnp.ones((1, IN_DIM))) # MNIST DUMMY INPUT
    tx = create_optimizer(flags)  # Initialize the optimizer
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the training step (forward pass + backward pass)
@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits, hidden_outputs = state.apply_fn(params, images)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        accuracy = (jnp.argmax(logits, axis=-1) == labels).mean()
        return loss, (accuracy, hidden_outputs)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (accuracy, hidden_outputs)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, (loss, accuracy), hidden_outputs


# Define the evaluation step (no gradient computation)
@jax.jit
def eval_step(state, images, labels):
    logits, _ = state.apply_fn(state.params, images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = (jnp.argmax(logits, axis=-1) == labels).mean()
    return loss, accuracy


# Define the main training loop
def train_model(flags):
    # Rng
    RNG = jrn.key(randint(-1000, 1000))
    
    # Initialize the model
    model = create_model(
        flags, 
        features=[flags.hidden_dim,] * (flags.num_layers - 1) + [NUM_CLASSES,]
    )
    # Initialize the model and optimmizer
    state = create_train_state(flags, model, rng=RNG)
    
    # MNIST dataloaders
    train_ds, test_ds = get_mnsit_dataloaders(batch_size=flags.batch_size)

    # initialize metrics trackers
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    l1_leaf_norm_history = []
    l1_output_norm_history = []
    
    # Train for a few epochs
    for epoch in tqdm(range(flags.epochs)):
        print(f"Epoch {epoch + 1}")
        
        # Training
        for step, (images, labels) in enumerate(train_ds):
            images, labels = jnp.array(images, dtype=jnp.float32), jnp.array(labels, dtype=jnp.int32)
            
            state, (loss, accuracy), hidden_outputs = train_step(state, images, labels)
            
            # logs metrics every 10 steps
            if step % 10 == 0:
                train_loss.append(loss)
                train_acc.append(accuracy)
                l1_leaf_norm_history.append(
                    jtu.tree_map(lambda leaf: jnp.linalg.norm(leaf, ord=1), state.params)
                )
                l1_output_norm_history.append(jnp.linalg.norm(hidden_outputs, ord=1, axis=-1))
                
        # Validation
        ev_loss = []
        ev_acc = []
        for step, (images, labels) in enumerate(test_ds):
            images, labels = jnp.array(images, dtype=jnp.float32), jnp.array(labels, dtype=jnp.int32)
            
            loss, accuracy = eval_step(state, images, labels)
            ev_loss.append(loss)
            ev_acc.append(accuracy)
        # compute mean on epoch
        val_loss.append(np.mean(ev_loss))
        val_acc.append(np.mean(ev_acc))
        
        # Print the training and validation loss and accuracy means for each epoch
        print(f"Training loss: {train_loss[-1]:.4f}")
        print(f"Training accuracy: {train_acc[-1]:.4f}")
        
        print(f"Validation loss: {val_loss[-1]:.4f}")
        print(f"Validation accuracy: {val_acc[-1]:.4f}")

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "l1_leaf_norm_history": l1_leaf_norm_history,
        "l1_output_norm_history": l1_output_norm_history,
    }
    return state, metrics