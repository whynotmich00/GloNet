from flax import nnx
from flax import linen as nn
from typing import Tuple, List
import jax.numpy as jnp


class MLP(nn.Module):
    features: Tuple[int]
    
    @nn.compact
    def __call__(self, x):
        # Initialize the output tensor with zeros
        layers_outputs = jnp.zeros(shape=(len(self.features) - 1, x.shape[0], self.features[0]))
        for i, o_dim in enumerate(self.features[:-1]):
            x = nn.Dense(features=o_dim)(x)
            x = nn.relu(x)
            layers_outputs = layers_outputs.at[i].set(x)
        
        logits = nn.Dense(features=self.features[-1])(x)
        return logits, layers_outputs[:-1]


# forward version
class GloNet_f(nnx.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, num_layers: int, *, rngs: nnx.Rngs):
        self.layers = []
        self.project = nnx.Linear(in_features=in_dim, out_features=h_dim, rngs=rngs)
        network_in_dim = (in_dim,) + (h_dim,) * (num_layers - 1)
        network_out_dim = (h_dim,) * (num_layers - 1) + (out_dim,) 
        
        for i_dim, o_dim in zip(network_in_dim, network_out_dim):
            self.layers.append(
                nnx.Linear(
                    in_features=i_dim,
                    out_features=o_dim,
                    rngs=rngs,
                )
            )

    def __call__(self, x):
        input_projection = self.project(x)
        # glonet has a residual connection between each layer in the network
        for i, layer in enumerate(self.layers): # iterate over all the layers up to the 
            
            if i != len(self.layers) - 1:
                x = nnx.relu(layer(x))
                x = x + input_projection
            else:
                x = layer(x)
        
        return x


# MAIN version
class GloNet(nn.Module):
    features: List[int]
    num_layers: int = None
    hidden_dim: int = None
    output_dim: int = None

    @nn.compact
    def __call__(self, x):
        # Initialize the output tensor with zeros
        layers_outputs = jnp.zeros(shape=(len(self.features) - 1, x.shape[0], self.features[0]))
        # glonet has a residual connection between each layer in the network
        for layer, o_dim in enumerate(self.features[:-1]):
            x = nn.Dense(features=o_dim)(x)
            x = nn.relu(x)
            layers_outputs = layers_outputs.at[layer].set(x)
        
        logits = nn.Dense(features=self.features[-1])(layers_outputs.sum(axis=0))
        
        # residual connection of all the output layer coming into the last output
        return logits, layers_outputs


# Michelangelo version
class GloNet_m(nnx.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, num_layers: int, *, rngs: nnx.Rngs):
        self.layers = []
        network_in_dim = (in_dim,) + (h_dim,) * (num_layers - 1)
        network_out_dim = (h_dim,) * (num_layers - 1) + (out_dim,) 
        
        for i_dim, o_dim in zip(network_in_dim, network_out_dim):
            self.layers.append(
                nnx.Linear(
                    in_features=i_dim,
                    out_features=o_dim,
                    rngs=rngs,
                )
            )

    def __call__(self, x):
        output_list = []
        # glonet has a residual connection between each layer in the network
        for layer in self.layers[:-1]:      # iterate over all the layers up to the last
            inp = x
            x = nnx.relu(layer(inp))        # forward pass of the layer with trainable weigths
            x += inp                        # skip connection
            output_list.append(x)           # append for the total residual summation
        
        total_residual = sum(output_list) # sum all the previous layers output

        logits = self.layers[-1](total_residual)
        
        # residual connection of all the output layer coming into the last output
        return logits