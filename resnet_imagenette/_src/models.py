from flax import linen as nn
from typing import Tuple, List
import jax.numpy as jnp
import jax

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
    num_classes: int = 10
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
    blocks_features: Tuple[int] = (64, 128, 256, 512)
    blocks_per_group: Tuple[int] = (3, 4, 6, 3)
    kernel_size: Tuple[int] = (3, 3)
    stride: int = 1
    output_dim: int = 10
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x, training=None):
        # first layer (7x7) kernel
        x = nn.Conv(
            features=128,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            name="Conv_embed",
            kernel_init=jax.nn.initializers.kaiming_normal(),
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        
        # Initialize the intermediates output tensor with zeros
        layers_outputs_sum = jnp.zeros(shape=x.shape[:-1] + (self.blocks_features[0],)) # shape (batch_size, height, width, features)
        # glonet has a residual connection between each layer and the final output in the network
        for i, (features, num_blocks) in enumerate(
            zip(self.blocks_features, self.blocks_per_group)
        ):
            for j in range(num_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                use_projection = (j == 0 and (i > 0 or features != 64)) # projection should be used for the first block of the group but not for the first group
                x = nn.Conv(
                    features=features,
                    kernel_size=self.kernel_size,
                    kernel_init=jax.nn.initializers.kaiming_normal(),
                    strides=stride,
                    padding="SAME",
                    use_bias=False,
                )(x)
                x = nn.relu(x)
                x = nn.BatchNorm(use_running_average=not training)(x) if self.use_batch_norm else x    
                
                if use_projection:
                    layers_outputs_sum = nn.Conv(
                        features=features,
                        kernel_size=(1, 1),
                        kernel_init=jax.nn.initializers.kaiming_normal(),
                        strides=2,
                        padding="SAME",
                        use_bias=False,
                        name=f"Projection_Conv_{i}_{j}",
                        )(layers_outputs_sum)
                
                    layers_outputs_sum += x
        
        # global average pooling: NB i am not sure that this is the correct approach but i need to reduce the
        # output dimension
        # layers_outputs_sum = nn.max_pool(layers_outputs_sum, (3, 3), strides=(2, 2), padding="SAME") # original code uses max pooling but it is not necessary for CIFAR10
        # batch_size, height, width, channels = layers_outputs_sum.shape
        # layers_outputs_sum = layers_outputs_sum.reshape((batch_size, height * width * channels))  # assuming x has shape (batch_size, height, width, channels)
        layers_outputs_sum = jnp.mean(layers_outputs_sum, axis=(1, 2))
        
        logits = nn.Dense(
            features=self.output_dim,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            bias_init=jax.nn.initializers.zeros,
            name="Dense_final",
            )(layers_outputs_sum)
        
        # residual connection of all the output layer coming into the last output
        return logits
    
    

ResNet20 = partial(ResNet, resnetblock_per_group=(3, 4, 6, 3))
ResNet34 = partial(ResNet, resnetblock_per_group=(3, 4, 6, 3))
ResNet50 = partial(ResNet, resnetblock_per_group=(3, 4, 6, 3), use_bottleneck_block=True)
ResNet101 = partial(ResNet, resnetblock_per_group=(3, 4, 23, 3), use_bottleneck_block=True)
ResNet152 = partial(ResNet, resnetblock_per_group=(3, 8, 36, 3), use_bottleneck_block=True)

GloNet20 = partial(GloNet, blocks_per_group=(3, 4, 6, 3))
GloNet34 = partial(GloNet, blocks_per_group=(3, 10, 12, 3))
GloNet50 = partial(GloNet, blocks_per_group=(3, 4, 6, 3))
GloNet101 = partial(GloNet, blocks_per_group=(3, 4, 6, 3))
GloNet152 = partial(GloNet, blocks_per_group=(3, 4, 6, 3))