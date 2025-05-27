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

from _src.get_data import get_cifar10_dataloaders
from _src.utils import create_model, TrainState, create_optimizer

# Cifar10 specifics setting
IN_DIM = (32, 32, 3)  # Input dimension for CIFAR-10 images
OUT_DIM = 10

# Define a function to initialize the model and optimizer
def create_train_state(flags, model, rng):
    variables = model.init(rng, jnp.ones((1,) + IN_DIM), training=False) # CIFAR10 DUMMY INPUT
    # print the number of parameters
    print(f"Number of parameters: {jtu.tree_reduce(add, jtu.tree_map(jnp.size, variables["params"]))}")
    tx = create_optimizer(flags)  # Initialize the optimizer
    return TrainState.create(
        apply_fn=model.apply, 
        params=variables["params"], 
        tx=tx, 
        batch_stats=variables["batch_stats"] if "batch_stats" in variables else {}
    )


def loss_function(labels, logits, params, alpha=1e-5):
    cross_entropy = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    regularization = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    return cross_entropy + alpha * regularization  # Add regularization term to the loss
    

# Define the training step (forward pass + backward pass)
@partial(jax.jit, static_argnames=["criterion"])
def train_step(state, batch_images, batch_labels, criterion):
    def loss_fn(params):
        logits, batch_updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats}, batch_images,
            training=True, mutable=["batch_stats"]
        )
        loss = criterion(batch_labels, logits, params)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_labels)
        return loss, (acc, batch_updates)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (acc, batch_updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads).replace(batch_stats=batch_updates["batch_stats"])
    
    return state, (loss, acc)


# Define the evaluation step (no gradient computation)
@partial(jax.jit, static_argnames=["criterion"])
def eval_step(state, batch_images, batch_labels, criterion):
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, batch_images, training=False,
    )
    loss = criterion(batch_labels, logits, state.params)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_labels)
    return loss, acc


# Define the evaluation step (no gradient computation)
@partial(jax.jit, static_argnames=["criterion"])
def eval_step_with_intermediates(state, batch_images, batch_labels, criterion):
    logits, intermediates_outputs = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, batch_images,
        training=False, capture_intermediates=True, mutable=["intermediates"],
    )
    loss = criterion(batch_labels, logits, state.params)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_labels)
    return loss, acc, intermediates_outputs


# Define the main training loop
def train_model(flags):
    # Rng
    RNG = jrn.key(randint(-1000, 1000))
    # Initialize the model
    model = create_model(flags)
    # Initialize the model and optimmizer
    state = create_train_state(flags, model, rng=RNG)
    # SGEMM dataloaders
    train_ds, test_ds = get_cifar10_dataloaders(batch_size=flags.batch_size)

    criterion = partial(loss_function, alpha=flags.l2_reg_alpha)
    # initialize metrics trackers
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    l1_leaf_norm_history = []
    
    # Train for a few epochs
    for epoch in tqdm(range(flags.epochs)):
        print(f"Epoch {epoch + 1}")
        
        # Training
        for step, (batch_images, batch_labels) in enumerate(train_ds):
            batch_images = jnp.array(batch_images, dtype=jnp.float32)
            batch_labels = jnp.array(batch_labels, dtype=jnp.int32).squeeze()
            
            state, (loss, acc) = train_step(state, batch_images, batch_labels, criterion)

            # logs metrics every 10 steps, batch_size is 1024 (by default), so it is 10k samples
            if step % 10 == 0:
                train_loss.append(loss)
                train_acc.append(acc)
                
                l1_leaf_norm_history.append(
                    jtu.tree_map(lambda leaf: jnp.linalg.norm(leaf.ravel(), ord=1), state.params)
                )
                
        # Validation
        ev_loss = []
        ev_acc = []
        buffer_tree = None
        for step, (batch_images, batch_labels) in enumerate(test_ds):
            batch_images = jnp.array(batch_images, dtype=jnp.float32)
            batch_labels = jnp.array(batch_labels, dtype=jnp.int32).squeeze()
            
            if epoch == flags.epochs - 1:
                loss, acc, intermeditates_outputs = eval_step_with_intermediates(state, batch_images, batch_labels, criterion)
                ev_loss.append(loss)
                ev_acc.append(acc)
                
                # current batch l1 norms
                cur_l1_norms = jtu.tree_map(lambda leaf: jnp.linalg.norm(leaf.ravel(), ord=1), intermeditates_outputs["intermediates"])
                buffer_tree = jtu.tree_map(add, buffer_tree, cur_l1_norms) if buffer_tree is not None else cur_l1_norms
            
            else:
                loss, acc = eval_step(state, batch_images, batch_labels, criterion)
                ev_loss.append(loss)
                ev_acc.append(acc)
         
        # compute mean on epoch
        val_loss.append(np.mean(ev_loss))
        val_acc.append(np.mean(ev_acc))
        
        # Print the training and validation loss and accuracy means for each epoch
        print(f"Training loss: {train_loss[-1]:.4f}")
        print(f"Validation loss (mean on epoch): {val_loss[-1]:.4f}")
        print(f"Training accuracy: {train_acc[-1]:.4f}")
        print(f"Validation accuracy (mean on epoch): {val_acc[-1]:.4f}")

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "l1_leaf_norm_history": (l1_leaf_norm_history, {"ordered_keys": list(state.params.keys())}),
        "l1_intermediate_output_validation": jtu.tree_map(
            lambda leaf: leaf / (len(test_ds) * flags.batch_size), 
            buffer_tree,
        )
    }
    return state, metrics