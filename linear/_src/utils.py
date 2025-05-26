from flax import linen as nn
from flax.training import train_state
import optax
from typing import Any, List

from _src.models import GloNet_m, GloNet, GloNet_f, MLP


def create_model(
    flags, 
    *,
    features: List[int],
    ) -> nn.Module:
    
    if flags.model == "GloNet":
        return GloNet(
            features=features,
        )
    # elif flags.model == "GloNet_f":
    #     return GloNet_f(
    #         in_dim=in_dim,
    #         h_dim=flags.hidden_dimension,
    #         out_dim=out_dim,
    #         num_layers=flags.num_layers,
    #         rngs=rng,
    #     )
    # elif flags.model == "GloNet_m":
    #     return GloNet_m(
    #         in_dim=in_dim,
    #         hid_dim=flags.hidden_dimension,
    #         out_dim=out_dim,
    #         num_layers=flags.num_layers,
    #         rngs=rng,
    #     )
    elif flags.model == "MLP":
        return MLP(
            features=features,
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
