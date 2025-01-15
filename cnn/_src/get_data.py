import random
from tensorflow.keras.datasets import mnist
from tensorflow import newaxis
from tensorflow.data import Dataset, AUTOTUNE
import grain.python as grain

def get_mnsit_dataloaders(batch_size):
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the images to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Add a channel dimension for compatibility with convolutional layers
    x_train = x_train[..., newaxis]
    x_test = x_test[..., newaxis]
    
    train_dataset = Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    test_dataset = Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    
    return train_dataset, test_dataset


def get_mnsit_grain_dataloaders(batch_size):
    # TODO: implement grain dataloader grain.DataLoader() 
    seed = random.randint(0, 10_000)
    
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    
    n_samples_t, *input_features = x_train.shape
    n_samples_v, *_ = x_val.shape
    
    # assert n_samples_t % batch_size == 0, f"{batch_size=} must divide {n_samples_t=}"
    train_n_batches = n_samples_t // batch_size
    
    # assert n_samples_v % batch_size == 0, f"{batch_size=} must divide {n_samples_v=}"
    val_n_batches = n_samples_v // batch_size
    
    x_train = x_train.reshape(train_n_batches, batch_size, *input_features)
    y_train = y_train.reshape(train_n_batches, batch_size, 1)
    x_val = x_val.reshape(val_n_batches, batch_size, *input_features)
    y_val = y_val.reshape(val_n_batches, batch_size, 1)
    
    train_index_sampler = grain.IndexSampler(num_records=x_train.shape[0],
                                            num_epochs=1,
                                            shard_options=grain.ShardOptions(
                                                shard_index=0, shard_count=1, drop_remainder=True),
                                            shuffle=True,
                                            seed=seed)
        
    val_index_sampler = grain.IndexSampler(num_records=x_val.shape[0],
                                        num_epochs=1,
                                        shard_options=grain.ShardOptions(
                                            shard_index=0, shard_count=1, drop_remainder=True),
                                        shuffle=True,
                                        seed=seed)
    
    train_dl = grain.DataLoader(data_source = (x_train, y_train),
                                sampler = train_index_sampler)
    
    val_dl = grain.DataLoader(data_source = (x_val, y_val),
                            sampler = val_index_sampler)
    
    return train_dl, val_dl