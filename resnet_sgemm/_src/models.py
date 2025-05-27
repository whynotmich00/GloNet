from flax import linen as nn
from typing import Tuple, List
import jax
import jax.numpy as jnp

from functools import partial


class ResNetDenseBlock(nn.Module):
    features: int
    use_projection: bool = False
    use_residual: bool = True
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        h = nn.Dense(
            features=self.features,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            bias_init=jax.nn.initializers.zeros,
        )(x)
        h = nn.BatchNorm(use_running_average=not training)(h) if self.use_batch_norm else h
        h = nn.relu(h)
        h = nn.Dense(
            features=self.features,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            bias_init=jax.nn.initializers.zeros,
        )(h)
        h = nn.BatchNorm(use_running_average=not training)(h) if self.use_batch_norm else h
        
        # x must be projected in a bigger space for the first resnet block of a group otherwise we can't compute elemetwise addition
        if self.use_projection and self.use_residual:
            x = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.kaiming_normal(),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        
        # residual connection
        if self.use_residual:
            h = h + x  
        
        return h


class ResNetBottleneckBlock(nn.Module):
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    stride: int = 1
    use_projection: bool = False
    use_residual: bool = True
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        bottleneck_channels = self.features // 4
        h = nn.Conv(
            features=bottleneck_channels,
            kernel_size=(1, 1),
            strides=self.stride,
            padding="VALID",
            use_bias=False,
        )(x)
        h = nn.BatchNorm(use_running_average=not training)(h) if self.use_batch_norm else h
        h = nn.relu(h)
        h = nn.Conv(
            features=bottleneck_channels,
            kernel_size=(3, 3),
            strides=1,
            padding="SAME",
            use_bias=False,
        )(h)
        h = nn.BatchNorm(use_running_average=not training)(h) if self.use_batch_norm else h
        h = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=1,
            padding="SAME",
            use_bias=False,
        )(h)
        
        # x must be projected in a bigger space for the first resnet block of a group otherwise we can't compute elemetwise addition
        if self.use_projection and self.use_residual:
            x = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=2,
            padding="SAME",
            use_bias=False,
            )(x)
        
        # residual connection
        if self.use_residual:
            h = h + x
             
        
        return h



class ResNet(nn.Module):
    resnet_blocks: int
    output_dim: int = 1
    features: int = 16
    use_residual: bool = True
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # first layer embed in R^16
        h = nn.Dense(
            features=16,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            bias_init=jax.nn.initializers.zeros,
        )(x)
        
        h = nn.BatchNorm(name="BatchNorm_0", use_running_average=not training)(h) if self.use_batch_norm else h
        h = nn.relu(h)
        
        # ResNetBlocks
        for block in range(self.resnet_blocks):
            use_projection = False # projection should be used for the first block of the group but not for the first group
                
            h = ResNetDenseBlock(
                    features=self.features,
                    use_projection=use_projection,
                    use_residual=self.use_residual,
                    use_batch_norm=self.use_batch_norm,
                    name=f"resnet_block{block}",
                )(h, training=training)
        
        h = nn.BatchNorm(name="BatchNorm_final", use_running_average=not training)(h) if self.use_batch_norm else h
        h = nn.relu(h)
        
        # final layers (1 out dim for regression)
        logits = nn.Dense(
            self.output_dim,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            bias_init=jax.nn.initializers.zeros,
            name="Dense_final",
            )(h)
        
        return logits


# MAIN version
class GloNet(nn.Module):
    features: int = 16
    num_layers: int = 4
    output_dim: int = 1

    @nn.compact
    def __call__(self, x, training = None):
        # Initialize the cumulative sum tensor with zeros
        layers_outputs_sum = jnp.zeros((x.shape[0], self.features))  # shape (batch_size, features)
        # glonet has a residual connection between each layer in the network
        for i in range(self.num_layers - 1):
            x = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.kaiming_normal(),
                bias_init=jax.nn.initializers.zeros,
                name=f"Dense_{i}"
            )(x)
            x = nn.relu(x)
            layers_outputs_sum += x  # accumulate outputs of each layer
        
        # final layer
        predictions = nn.Dense(
            features=self.output_dim,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            bias_init=jax.nn.initializers.zeros,
            name=f"Dense_{self.num_layers - 1}"
        )(layers_outputs_sum)
        
        return predictions


class MLP(nn.Module):
    features: List[int]
    num_layers: int = 4
    output_dim: int = 1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        h = x
        for i in range(self.num_layers - 1):
            h = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.kaiming_normal(),
                bias_init=jax.nn.initializers.zeros,
                name=f"dense_{i}"
            )(h)
            h = nn.relu(h)
        
        logits = nn.Dense(
            features=self.output_dim,
            name="output"
        )(h)
        
        return logits



ResNet10 = partial(ResNet, resnet_blocks=5)            # 5 blocks, each with 2 Dense layers
ResNet24 = partial(ResNet, resnet_blocks=12)           # 12 blocks, each with 2 Dense layers
ResNet50 = partial(ResNet, resnet_blocks=25)           # 25 blocks, each with 2 Dense layers
ResNet100 = partial(ResNet, resnet_blocks=50)          # 50 blocks, each with 2 Dense layers
ResNet200 = partial(ResNet, resnet_blocks=100)         # 100 blocks, each with 2 Dense layers


GloNet10 = partial(GloNet, num_layers=12)
GloNet24 = partial(GloNet, num_layers=26)
GloNet50 = partial(GloNet, num_layers=52)
GloNet100 = partial(GloNet, num_layers=102)
GloNet200 = partial(GloNet, num_layers=202)