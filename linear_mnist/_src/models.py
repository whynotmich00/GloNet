from flax import linen as nn
import jax.numpy as jnp
import jax


class MLP(nn.Module):
    features: int
    num_layers: int = None
    output_dim: int = None
    
    @nn.compact
    def __call__(self, x):
        # Initialize the output tensor with zeros
        for layer in range(self.num_layers - 1):
            x = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.kaiming_normal(),
                bias_init=jax.nn.initializers.zeros,
                name=f"Dense_{layer}",
                )(x)
            x = nn.relu(x)
        
        logits = nn.Dense(
            features=self.output_dim,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            bias_init=jax.nn.initializers.zeros,
            name=f"Dense_{self.num_layers - 1}",
            )(x)
        
        return logits



# MAIN version
class GloNet(nn.Module):
    features: int
    num_layers: int = None
    output_dim: int = None

    @nn.compact
    def __call__(self, x):
        # Initialize the output tensor with zeros
        layers_outputs_sum = jnp.zeros((x.shape[0], self.features))
        # glonet has a residual connection between each layer in the network
        for layer in range(self.num_layers - 1):
            x = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.kaiming_normal(),
                bias_init=jax.nn.initializers.zeros,
                name=f"Dense_{layer}",
            )(x)
            x = nn.relu(x)
            layers_outputs_sum += x
        
        logits = nn.Dense(
            features=self.output_dim,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            bias_init=jax.nn.initializers.zeros,
            name=f"Dense_{self.num_layers - 1}",
        )(layers_outputs_sum)
        
        # residual connection of all the output layer coming into the last output
        return logits

