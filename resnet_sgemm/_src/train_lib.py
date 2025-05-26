import jax
import jax.numpy as jnp
import jax.random as jrn
import jax.tree_util as jtu
import optax
from tqdm import tqdm
from random import randint
import numpy as np
from operator import add
from functools import partial

from _src.get_data import get_sgemm_dataloaders
from _src.utils import create_model, TrainState, create_optimizer

# Imagenette specifics setting
IN_DIM = 14
OUT_DIM = 1

# Define a function to initialize the model and optimizer
def create_train_state(flags, model, rng):
    variables = model.init(rng, jnp.ones((1, IN_DIM)), training=False) # SGEMM DUMMY INPUT
    # print the number of parameters
    print(f"Number of parameters: {jtu.tree_reduce(add, jtu.tree_map(jnp.size, variables["params"]))}")
    tx = create_optimizer(flags)  # Initialize the optimizer
    return TrainState.create(
        apply_fn=model.apply, 
        params=variables["params"], 
        tx=tx, 
        batch_stats=variables["batch_stats"] if "batch_stats" in variables else {}
    )


def loss_function(batch_y, preds, params, alpha=1e-5):
    mse = optax.squared_error(preds, batch_y).mean()
    regularization = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    return mse + alpha * regularization  # Add regularization term to the loss
    

# Define the training step (forward pass + backward pass)
@partial(jax.jit, static_argnames=["criterion"])
def train_step(state, batch_x, batch_y, criterion):
    def loss_fn(params):
        preds, batch_updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats}, batch_x,
            training=True, mutable=["batch_stats"]
        )
        loss = criterion(batch_y, preds, params)
        return loss, batch_updates
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, batch_updates), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads).replace(batch_stats=batch_updates["batch_stats"])
    
    return state, loss


# Define the evaluation step (no gradient computation)
@partial(jax.jit, static_argnames=["criterion"])
def eval_step(state, batch_x, batch_y, criterion):
    preds = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, batch_x,
        training=False, capture_intermediates=True, mutable=["intermediates"],
    )
    loss = criterion(batch_y, preds, state.params)
    return loss


# Define the evaluation step (no gradient computation)
@partial(jax.jit, static_argnames=["criterion"])
def eval_step_with_intermediates(state, batch_x, batch_y, criterion):
    preds, intermediates_outputs = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, batch_x,
        training=False, capture_intermediates=True, mutable=["intermediates"],
    )
    loss = criterion(batch_y, preds, state.params)
    return loss, intermediates_outputs


# Define the main training loop
def train_model(flags):
    # Rng
    RNG = jrn.key(randint(-1000, 1000))
    # Initialize the model
    model = create_model(flags)
    # Initialize the model and optimmizer
    state = create_train_state(flags, model, rng=RNG)
    # SGEMM dataloaders
    train_ds, test_ds = get_sgemm_dataloaders(batch_size=flags.batch_size)

    criterion = partial(loss_function, alpha=flags.l2_reg_alpha)
    # initialize metrics trackers
    train_loss = []
    val_loss = []
    l1_leaf_norm_history = []
    l1_intermediate_output_last_epoch_validation = []
    
    # Train for a few epochs
    for epoch in tqdm(range(flags.epochs)):
        print(f"Epoch {epoch + 1}")
        
        # Training
        for step, (batch_x, batch_y) in enumerate(train_ds):
            batch_x = jnp.array(batch_x, dtype=jnp.float32)
            batch_y = jnp.log(jnp.array(batch_y, dtype=jnp.float32).mean(axis=-1, keepdims=True))
            
            state, loss = train_step(state, batch_x, batch_y, criterion)

            # logs metrics every 10 steps, batch_size is 1024 (by default), so it is 10k samples
            if step % 10 == 0:
                train_loss.append(loss)
                
                l1_leaf_norm_history.append(
                    jtu.tree_map(lambda leaf: jnp.linalg.norm(leaf.ravel(), ord=1), state.params)
                )
                
        # Validation
        ev_loss = []
        for step, (batch_x, batch_y) in enumerate(test_ds):
            batch_x = jnp.array(batch_x, dtype=jnp.float32)
            batch_y = jnp.log(jnp.array(batch_y, dtype=jnp.float32).mean(axis=-1, keepdims=True))
            
            if epoch == flags.epochs - 1:
                loss, intermeditates_outputs = eval_step_with_intermediates(state, batch_x, batch_y, criterion)
                ev_loss.append(loss)
                l1_intermediate_output_last_epoch_validation.append(intermeditates_outputs["intermediates"])
            
            else:
                loss = eval_step(state, batch_x, batch_y, criterion)
                ev_loss.append(loss)
         
        # compute mean on epoch
        val_loss.append(np.mean(ev_loss))
        # l1_output_norm_history.append(jnp.array(ev_l1_norm).sum(axis=0) / (len(test_ds) * flags.batch_size))
        
        # Print the training and validation loss and accuracy means for each epoch
        print(f"Training loss: {train_loss[-1]:.4f}")
        print(f"Validation loss: {val_loss[-1]:.4f}")

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "l1_leaf_norm_history": (l1_leaf_norm_history, {"ordered_keys": list(state.params.keys())}),
        "l1_intermediate_output_validation": (
            l1_intermediate_output_last_epoch_validation, {"len_testset": len(test_ds), "batch_size": flags.batch_size}
            ),
    }
    return state, metrics