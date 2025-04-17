from flax import linen as nn
from flax import nnx
from typing import Tuple, List


class CNN(nn.Module):
    """A simple CNN model."""
    features: Tuple
    kernel_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features[0], kernel_size=self.kernel_size)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=self.features[1], kernel_size=self.kernel_size)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=self.features[2])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features[3])(x)
        return x


class CNN_nnx(nnx.Module):
    """A simple CNN model."""
    def __init__(self, features: tuple[int, ...], kernel_size: tuple[int, int], rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(in_features=3, out_features=features[0], kernel_size=kernel_size, rngs=rngs)
        self.conv1 = nnx.Conv(in_features=features[0], out_features=features[1], kernel_size=kernel_size, rngs=rngs)
        self.dense0 = nnx.Linear(in_features=features[1], out_features=features[2], rngs=rngs)
        self.dense1 = nnx.Linear(in_features=features[2], out_features=features[3], rngs=rngs)
    
    def __call__(self, x):
        x = self.conv0(x)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.dense0(x)
        x = nnx.relu(x)
        x = self.dense1(x)
        return x