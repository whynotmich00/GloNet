"""
SGEMM GPU Kernel Performance Dataset Setup and Preparation
Dataset from UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance
"""

import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import zipfile
from pathlib import Path
import json
from tensorflow.data import Dataset, AUTOTUNE


def download_sgemm_dataset(data_dir="./data"):
    """
    Download SGEMM GPU kernel performance dataset from UCI
    """
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # UCI dataset URL
    url = "https://archive.ics.uci.edu/static/public/440/sgemm+gpu+kernel+performance.zip"
    zip_path = os.path.join(data_dir, "sgemm_product.zip")
    
    print("Downloading SGEMM dataset...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded dataset to {zip_path}")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print(f"Extracted dataset to {data_dir}")
        
        # Remove zip file
        os.remove(zip_path)
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_and_prepare_data(data_dir="./data"):
    """
    Load and prepare the SGEMM dataset
    """
    # Try different possible filenames
    possible_files = [
        "sgemm_product.csv",
        "sgemm_product_dataset/sgemm_product.csv",
        "sgemm_product_dataset.csv"
    ]
    
    csv_path = None
    for filename in possible_files:
        full_path = os.path.join(data_dir, filename)
        if os.path.exists(full_path):
            csv_path = full_path
            break
    
    if csv_path is None:
        # List available files
        print("Available files in data directory:")
        for item in Path(data_dir).rglob("*"):
            if item.is_file():
                print(f"  {item}")
        raise FileNotFoundError("Could not find SGEMM dataset CSV file")
    
    print(f"Loading data from: {csv_path}")
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def explore_dataset(df):
    """
    Explore the SGEMM dataset
    """
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    print(f"\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nColumn Information:")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        print(f"{i+1:2d}. {col:20s} | {str(dtype):10s} | {unique_count:6d} unique | {null_count:6d} nulls")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values:")
        print(missing[missing > 0])
    else:
        print(f"\nNo missing values found!")
    
    return df

def preprocess_sgemm_data(df, target_cols=None):
    """
    Preprocess SGEMM dataset for machine learning
    
    The dataset typically contains:
    - Matrix dimensions (MWG, NWG, KWG, etc.)
    - GPU kernel parameters 
    - Performance metrics (Run1, Run2, Run3, Run4 - execution times)
    """
    
    # Make a copy
    data = df.copy()
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Identify likely target columns (performance metrics)
    if target_cols is None:
        # Look for columns that might be performance metrics
        performance_cols = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['run', 'time', 'performance', 'speed']):
                performance_cols.append(col)
        
        if performance_cols:
            print(f"Detected performance columns: {performance_cols}")
            target_cols = performance_cols
        else:
            # If no obvious performance columns, use last few columns
            target_cols = data.columns[-4:].tolist()
            print(f"Using last 4 columns as targets: {target_cols}")
    
    # Separate features and targets
    feature_cols = [col for col in data.columns if col not in target_cols]
    
    X = data[feature_cols].copy()
    y = data[target_cols].copy()
    
    print(f"\nFeatures: {len(feature_cols)} columns")
    print(f"Targets: {len(target_cols)} columns")
    
    # Handle categorical variables if any
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Categorical columns found: {categorical_cols}")
        # Apply label encoding to categorical columns
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
        print("Applied label encoding to categorical columns")
    
    # Handle any remaining missing values
    if X.isnull().any().any():
        print("Filling missing values with median...")
        X = X.fillna(X.median())
    
    if y.isnull().any().any():
        print("Filling missing target values with median...")
        y = y.fillna(y.median())
    
    print(f"\nFinal shapes:")
    print(f"X (features): {X.shape}")
    print(f"y (targets): {y.shape}")
    
    return X, y, feature_cols, target_cols

def create_train_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/validation/test splits
    """
    print("\n" + "="*50)
    print("CREATING DATA SPLITS")
    print("="*50)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # Second split: separate train and validation from remaining data  
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Val set:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")  
    print(f"Test set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler
    """
    print("\n" + "="*50)
    print("FEATURE SCALING")
    print("="*50)
    
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    print(f"Feature means: {scaler.mean_[:5]}...")  # Show first 5
    print(f"Feature stds:  {scaler.scale_[:5]}...")  # Show first 5
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                       feature_cols, target_cols, scaler, data_dir="./data"):
    """
    Save processed data for later use
    """
    print("\n" + "="*50)
    print("SAVING PROCESSED DATA")
    print("="*50)
    
    processed_dir = os.path.join(data_dir, "processed")
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'shapes': {
            'X_train': X_train.shape,
            'X_val': X_val.shape, 
            'X_test': X_test.shape,
            'y_train': y_train.shape,
            'y_val': y_val.shape,
            'y_test': y_test.shape
        }
    }
    
    import json
    with open(os.path.join(processed_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, os.path.join(processed_dir, "scaler.pkl"))
    
    print(f"Saved processed data to: {processed_dir}")
    print("Files saved:")
    for item in Path(processed_dir).iterdir():
        print(f"  {item.name}")

def main():
    """
    Main function to download and prepare SGEMM dataset
    """
    print("SGEMM GPU Kernel Performance Dataset Setup")
    print("=" * 60)
    
    # Step 1: Install dependencies (informational)
    # install_dependencies()
    
    # Step 2: Download dataset
    success = download_sgemm_dataset()
    if not success:
        print("Failed to download dataset. Please download manually from:")
        print("https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance")
        return
    
    # Step 3: Load and explore data
    df = load_and_prepare_data()
    df = explore_dataset(df)
    
    # Step 4: Preprocess data
    X, y, feature_cols, target_cols = preprocess_sgemm_data(df)
    
    # Step 5: Create train/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
    
    # Step 6: Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    # Step 7: Save processed data
    save_processed_data(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        feature_cols, target_cols, scaler
    )
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("You can now use the processed data for training neural networks.")
    print("\nNext steps:")
    print("1. Load the processed data:")
    print("   X_train = np.load('data/processed/X_train.npy')")
    print("   y_train = np.load('data/processed/y_train.npy')")
    print("2. Train your Flax model with batch normalization!")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

if __name__ == "__main__":
    # Run the main setup function
    data = main()