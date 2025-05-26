import numpy as np
import json
from tensorflow.data import Dataset, AUTOTUNE

def load_processed_data(data_dir="./data/processed"):
    """Load the processed SGEMM data"""
    
    X_train = np.load(f"{data_dir}/X_train.npy")
    X_val = np.load(f"{data_dir}/X_val.npy") 
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    
    # Load metadata
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  Features: {len(metadata['feature_cols'])}")
    print(f"  Targets: {len(metadata['target_cols'])}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test), metadata

def get_sgemm_dataloaders(batch_size: int = 32):
    (X_train, X_val, X_test, y_train, y_val, y_test), metadata = load_processed_data()
    
    train_dataset = Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    test_dataset = Dataset.from_tensor_slices((X_val, y_val)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    return train_dataset, test_dataset
