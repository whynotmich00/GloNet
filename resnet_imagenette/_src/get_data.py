from tensorflow.keras.datasets import mnist
from tensorflow.data import Dataset, AUTOTUNE
# Imagenette dataloader
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np


def get_mnsit_dataloaders(batch_size: int):
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the images to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    train_dataset = Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    test_dataset = Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1024).batch(batch_size=batch_size).prefetch(AUTOTUNE)
    
    return train_dataset, test_dataset


def get_imagenette_dataloaders(
    data_dir, 
    batch_size: int = 32, 
    image_size: int = 224
    ):
    """
    Create training and validation datasets for Imagenette using TensorFlow.
    Works with the standard Imagenette2 directory structure with separate train and val folders.
    
    Args:
        data_dir (str): Path to Imagenette dataset directory
        batch_size (int): Batch size for training and validation
        image_size (int): Target image size for model input
        
    Returns:
        tuple: (train_dataset, val_dataset, class_names)
    """
    # ImageNet mean and standard deviation for normalization
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    
    # Define preprocessing function for training data
    def preprocess_train(image, label):
        # Resize and crop
        image = tf.image.resize(image, [image_size + 32, image_size + 32])
        image = tf.image.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        
        # Random color adjustments
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        # Normalize to [0,1] then apply ImageNet normalization
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - mean) / std
        
        return image, label
    
    # Define preprocessing function for validation data
    def preprocess_val(image, label):
        image = tf.image.resize(image, [image_size + 32, image_size + 32])
        image = tf.image.central_crop(image, image_size / (image_size + 32))
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - mean) / std
        return image, label
    
    # Function to load and process images from a directory
    def process_directory(directory_path, is_training=True):
        print(f"dataset loaded from {os.path.abspath(directory_path)}")
        # Get the class folders
        class_dirs = sorted([d for d in os.listdir(directory_path) 
                            if os.path.isdir(os.path.join(directory_path, d))])
        
        # Map class folders to indices
        class_to_idx = {class_name: i for i, class_name in enumerate(class_dirs)}
        
        # Build the dataset
        all_images = []
        all_labels = []
        
        for class_name in class_dirs:
            class_path = os.path.join(directory_path, class_name)
            class_idx = class_to_idx[class_name]
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    all_images.append(img_path)
                    all_labels.append(class_idx)
        
        # Convert to TensorFlow tensors
        all_images = tf.constant(all_images)
        all_labels = tf.constant(all_labels)
        
        # Create the dataset
        dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
        
        # Function to load the image files
        def load_image(file_path, label):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            return img, label
        
        # Apply the loading function
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Apply preprocessing based on whether it's training or validation
        preprocess_fn = preprocess_train if is_training else preprocess_val
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Shuffle if training
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(all_images))
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset, class_dirs
    
    # Process training and validation directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_ds, class_dirs = process_directory(train_dir, is_training=True)
    val_ds, _ = process_directory(val_dir, is_training=False)
    
    # Map folder names to human-readable class names
    imagenette_classes = {
        'n01440764': 'tench',
        'n02102040': 'springer_spaniel',
        'n02979186': 'cassette_player',
        'n03000684': 'chain_saw',
        'n03028079': 'church',
        'n03394916': 'French_horn',
        'n03417042': 'garbage_truck',
        'n03425413': 'gas_pump',
        'n03445777': 'golf_ball',
        'n03888257': 'parachute'
    }
    
    class_names = [imagenette_classes.get(class_name, class_name) for class_name in class_dirs]
    
    return train_ds, val_ds, class_names


# Example usage:
# train_ds, val_ds, class_names = create_imagenette_dataset('/path/to/imagenette2')

# Function to visualize some images from the dataset
def visualize_dataset(dataset, class_names, n=9):
    plt.figure(figsize=(12, 12))
    
    # Get a batch from the dataset
    images, labels = next(iter(dataset))
    
    # Denormalize the images for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = images.numpy() * std + mean
    images = np.clip(images, 0, 1)
    
    # Plot the images
    for i in range(min(n, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i].numpy()])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()