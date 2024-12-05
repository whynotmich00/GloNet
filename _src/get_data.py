from tensorflow.keras.datasets import mnist
from tensorflow import newaxis
from tensorflow.data import Dataset, AUTOTUNE

from functools import reduce

def get_number_of_parameters(params):
    total = []
    for layer in params["params"].keys():
        for par in params["params"][layer].items():
            total.append((layer, reduce(lambda x, y: x * y, par[1].shape)))
    print(total)

def get_mnsit_dataloaders(batch_size, model_type="CNN"):
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the images to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Add a channel dimension for compatibility with convolutional layers
    x_train = x_train[..., newaxis]
    x_test = x_test[..., newaxis]
    
    if model_type == "MLP":
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    
    train_dataset = Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    test_dataset = Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    
    return train_dataset, test_dataset