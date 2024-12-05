import jax.numpy as jnp
import jax
import flax
from flax import linen as nn
from typing import Tuple, List


class CNN(nn.Module):
    """A simple CNN model."""
    features_shapes: Tuple
    kernel_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features_shapes[0], kernel_size=self.kernel_size)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=self.features_shapes[1], kernel_size=self.kernel_size)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=self.features_shapes[2])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features_shapes[3])(x)
        return x



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