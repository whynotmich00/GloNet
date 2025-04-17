from flax import nnx
from flax import linen as nn
from typing import Tuple, List


class MLP(nn.Module):
    features_shapes: Tuple[int]
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features_shapes[0])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features_shapes[1])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features_shapes[2])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features_shapes[3])(x)
        return x


class MLP_nnx(nnx.Module):
    """A simple CNN model."""
    def __init__(self, features: tuple[int, ...], rngs: nnx.Rngs):
        self.dense0 = nnx.Linear(features=features[0], rngs=rngs)
        self.dense1 = nnx.Linear(features=features[1], rngs=rngs)
        self.dense2 = nnx.Linear(features=features[2], rngs=rngs)
        self.dense3 = nnx.Linear(features=features[3], rngs=rngs)
    
    def __call__(self, x):
        x = self.dense0(x)
        x = nnx.relu(x)
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        x = nnx.relu(x)
        x = self.dense3(x)
        return x