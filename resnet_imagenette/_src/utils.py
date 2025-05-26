from flax import linen as nn
from flax.training import train_state
import optax
from typing import Any, List

from _src.models import GloNet, ResNet50, ResNet101, ResNet152


def create_model(
    flags, 
    *,
    features: List[int],
    ) -> nn.Module:
    
    if flags.model == "GloNet":
        return GloNet(features=features)
    elif flags.model == "ResNet50":
        return ResNet50()
    elif flags.model == "ResNet101":
        return ResNet101()
    elif flags.model == "ResNet152":
        return ResNet152()
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
