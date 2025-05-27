from tensorflow.keras.datasets import cifar10
from tensorflow.data import Dataset, AUTOTUNE
# Imagenette dataloader


def get_cifar10_dataloaders(batch_size: int):
    # Load the CIFAR10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the images to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    train_dataset = Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    test_dataset = Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    
    return train_dataset, test_dataset
