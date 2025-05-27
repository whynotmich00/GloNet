from flax import linen as nn
from flax.training import train_state
import optax
from typing import Any, Tuple

from _src.models import GloNet, MLP
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


def create_model(flags) -> nn.Module:
    
    if flags.model == "GloNet":
        return GloNet(
            features=flags.features,
            num_layers=flags.num_layers,
            output_dim=10,
        )
    elif flags.model == "MLP":
        return MLP(
            features=flags.features,
            num_layers=flags.num_layers,
            output_dim=10,
        )
    else:
        raise ValueError("Model not implemented.")

# Define the training state class to keep track of parameters and optimizer
class TrainState(train_state.TrainState):
    batch_stats: Any = None


# Define a function for creating an optimizer
def create_optimizer(flags):
    if flags.optimizer == "SGD": 
        return optax.sgd(learning_rate=flags.learning_rate, momentum=flags.momentum)
    elif flags.optimizer == "ADAM":
        return optax.adam(learning_rate=flags.learning_rate)
    else:
        raise ValueError(f"Optimizer {flags.optimizer} not implemented")


def compute_l1_norms_flattened(intermediates):
    intermediates.pop("__call__", None) # pop the final output (is repeated in the intermediates)
    # compute l1 norms for each output
    inter_l1_norm = jtu.tree_map(lambda x: jnp.linalg.norm(x.ravel(), ord=1), intermediates)
    
    ordered_out_values = []
    for k, v in inter_l1_norm.items():
        if "resnet_block" in k:
            for k_block, v_block in v.items():
                if "__call__" in k_block:
                    ordered_out_values.append(v["__call__"][0])
                else:
                    ordered_out_values.append(v_block["__call__"][0])
        else:
            ordered_out_values.append(v["__call__"][0])
    
    return ordered_out_values


def compute_l1_validation_mean_inter_outputs(intermediates_outputs: Tuple, test_ds_length: int, batch_size: int):
    # compute l1 norms for each batch intermediate output
    l1_inter_norms = np.array(
        [compute_l1_norms_flattened(intermediates) for intermediates in intermediates_outputs]
    )
    # compute mean over the validation set
    return l1_inter_norms.sum(axis=0) / (test_ds_length * batch_size)


def compute_l1_norms_leaves(norm_params, ordered_keys: list) -> list:
    ord_values = list(map(norm_params.get, ordered_keys))
    ordered_norm_params = {k: v for k, v in zip(ordered_keys, ord_values)}
    values = []
    for k, v in ordered_norm_params.items():
        if "bias" in v: # is a layer
            values.append(v["bias"] + v["kernel"] if "kernel" in v else v["bias"] + v["scale"])
        else: # is a block
            for layer in ["BatchNorm_0", "Conv_0", "BatchNorm_1", "Conv_1", "Conv_2", "Projection_Conv"]: # order of layers in block
                if layer in v:
                    vv = v[layer]
                    values.append(sum(vv.values()))
    
    return values