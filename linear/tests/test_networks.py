import pytest
import jax.numpy as jnp
from jax import random
from _src.Processors import CNN, MLP


@pytest.mark.parametrize("batch_size, height, width, in_channels, kernel_size, features_shapes, last_shape", [(1, 32, 32, 3, 4, (64, 128, 256, 10), 10)])
def test_CNN(batch_size, height, width, in_channels, kernel_size, features_shapes, last_shape):
    # get a sample of images
    images = jnp.ones((batch_size, height, width, in_channels))
    
    # initialize model
    model = CNN(kernel_size=(kernel_size,)*2, features_shapes=features_shapes)
    
    # initialize parameters
    params = model.init(random.PRNGKey(0), jnp.ones(shape=(1, height, width, in_channels)))
    
    # forward pass
    output = model.apply(params, images)
    
    assert output.shape == (batch_size, last_shape), "model output shape not equal to the last layer output shape"


@pytest.mark.parametrize("batch_size, height, width, in_channels, features_shapes, last_shape", [(10, 32, 32, 3, (64, 128, 256, 10), 10)])
def test_MLP(batch_size, height, width, in_channels, features_shapes, last_shape):
    # get a sample of images
    images = jnp.ones((batch_size, height, width, in_channels))
    
    images = images.reshape((batch_size, -1))
    
    # initialize model
    model = MLP(features_shapes=features_shapes)
    
    # initialize parameters
    params = model.init(random.PRNGKey(0), jnp.ones(shape=(1, height, width, in_channels)).reshape(1, -1))
    
    # forward pass
    output = model.apply(params, images)
    
    assert output.shape == (batch_size, last_shape), "model output shape not equal to the last layer output shape"



if __name__ == "__main__":
    pytest.main()