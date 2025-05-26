from flax import linen as nn
from typing import Tuple, List
import jax.numpy as jnp

from functools import partial


class ResNetBlock(nn.Module):
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    stride: int = 1
    use_projection: bool = False
    use_residual: bool = True
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        h = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="SAME",
            use_bias=False,
        )(x)
        h = nn.BatchNorm(use_running_average=not training)(h) if self.use_batch_norm else h
        h = nn.relu(h)
        h = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=1,
            padding="SAME",
            use_bias=False,
        )(h)
        h = nn.BatchNorm(use_running_average=not training)(h) if self.use_batch_norm else h
        
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
    resnetblock_per_group: Tuple[int, int, int, int]
    num_classes: int = 1000
    block_features: Tuple[int, ...] = (64, 128, 256, 512)
    kernel_size: Tuple[int, int] = (3, 3)
    use_bottleneck_block: bool = False
    use_residual: bool = True
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # first layer (7x7) kernel
        h = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(x)
        
        h = nn.BatchNorm(name="bn1", use_running_average=not training)(h) if self.use_batch_norm else h
        h = nn.relu(h)
        
        h = nn.max_pool(h, (3, 3), strides=(2, 2), padding="SAME")
        
        # ResNetBlocks
        for i, (features, num_blocks) in enumerate(
            zip(self.block_features, self.resnetblock_per_group)
        ):
            for j in range(num_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                use_projection = (j == 0 and (i > 0 or features != 64)) # projection should be used for the first block of the group but not for the first group
                
                if self.use_bottleneck_block:
                    h = ResNetBlock(
                        features=features,
                        kernel_size=self.kernel_size,
                        stride=stride,
                        use_projection=use_projection,
                        use_residual=self.use_residual,
                        use_batch_norm=self.use_batch_norm,
                        name=f"resnetblock{j}_group{i}",
                    )(h, training=training)
                
                else:
                    h = ResNetBottleneckBlock(
                        features=features,
                        kernel_size=self.kernel_size,
                        stride=stride,
                        use_projection=use_projection,
                        use_residual=self.use_residual,
                        use_batch_norm=self.use_batch_norm,
                        name=f"resnet_bottleneckblock{j}_group{i}"
                    )(h, training=training)
        
        h = nn.BatchNorm(name="bn_final", use_running_average=not training)(h) if self.use_batch_norm else h
        h = nn.relu(h)
        
        # global average pooling
        h = jnp.mean(h, axis=(1, 2))
        
        # final layers (1000 classes)
        logits = nn.Dense(
            self.num_classes,
            name="fc",
            )(h)
        
        return logits
        


# MAIN version
class GloNet(nn.Module):
    features: List[int]
    num_layers: int = None
    hidden_dim: int = None
    output_dim: int = None

    @nn.compact
    def __call__(self, x, training=None):
        # Initialize the output tensor with zeros
        layers_outputs = []
        # glonet has a residual connection between each layer in the network
        for layer, o_dim in enumerate(self.features[:-1]):
            x = nn.Conv(
                features=o_dim,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding="SAME",
                use_bias=False,
            )(x)
            x = nn.relu(x)
            layers_outputs.append(x)
        
        logits = nn.Dense(features=self.features[-1])(
            jnp.array(layers_outputs).sum(axis=0) # axis 0 is the one for the accumulation of outputs
            )
        
        # residual connection of all the output layer coming into the last output
        return logits, layers_outputs


ResNet34 = partial(ResNet, resnetblock_per_group=(3, 4, 6, 3))
ResNet50 = partial(ResNet, resnetblock_per_group=(3, 4, 6, 3), use_bottleneck_block=True)
ResNet101 = partial(ResNet, resnetblock_per_group=(3, 4, 23, 3), use_bottleneck_block=True)
ResNet152 = partial(ResNet, resnetblock_per_group=(3, 8, 36, 3), use_bottleneck_block=True)
