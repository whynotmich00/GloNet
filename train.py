import argparse
from jax import devices

from _src.Processors import MLP, CNN
from _src.get_data import get_mnsit_dataloaders
from _src.utils_functions import train_model
from _src.config import Config
from _src.log_training import log_training


def parse_arguments():
    """
    Parse command-line arguments for model configuration.
    """
    parser = argparse.ArgumentParser(description="Configurable Machine Learning Model")
    
    # Model architecture
    parser.add_argument('--model', default="MLP",
                        help='Model architecture')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, 
                        help='Learning rate for optimization')
    parser.add_argument('--momentum', type=float, default=0.0, 
                        help='Momentum regularization')
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='Batch size for training')
    
    # Model architecture parameters
    parser.add_argument('--features', type=tuple, default=(32, 64, 256, 10), 
                        help='Features size for each layer')
    parser.add_argument('--kernel-size', type=int, default=4, 
                        help='Convolution kernel size')
    
    parser.add_argument("--track-metrics", type=bool, default=True,
                        help="Log losses and accuracy during training and validation")
        
    return parser.parse_args()



if __name__ == "__main__":
    
    # Parse command-line arguments
    args = parse_arguments()
    
    assert args.model in ["MLP", "CNN"], "model must be 'MLP' or 'CNN'"
    assert len(args.features) == 4, "model has 4 layers"
    assert args.features[-1] == 10, "Last output shape must be 10 for MNIST dataset"
    
    # Print configuration for verification
    config = Config().as_dict(args)
    config_training, config_model, _, config_optimizer = config.items()
    
    print("Model Configuration:")
    print(f"Network: {config_model["model"]}")
    print(f"Epochs: {config_training["epochs"]}")
    if config_model["model"] == "CNN": print(f"Kernel size: {config_model["kernel_size"]}")
    print(f"Batch Size: {config_training["batch_size"]}")
    print(f"Learning Rate: {config_optimizer["learning_rate"]}")
    print(f"Momentum: {config_optimizer["momentum"]}") if config_optimizer["optimizer"] == "SGD" else None
    print(f"Out Channels: {config_model["features_shapes"]}")
    print(f"Track metrics: {config["track_metrics"]}")
    
    # List all available devices
    devices = devices()
    print("Available devices:")
    for device in devices:
        print(device)
    
    # Initialize the model
    if config["model"] == "MLP":
        network = MLP(features_shapes=config_model["features_shapes"])
    elif config["model"] == "CNN":
        network = CNN(kernel_size=(config_model["kernel_size"],)*2, features_shapes=config_model["features_shapes"])
    
    
    # MNIST dataloaders
    train_ds, test_ds = get_mnsit_dataloaders(batch_size=config_training["batch_size"], model_type=config_model["model"])
    
    
    # Train the model
    state, loss_tracker, accuracy_tracker = train_model(
                                                        model=network,
                                                        train_ds=train_ds,
                                                        test_ds=test_ds,
                                                        config_optimizer=config_optimizer,
                                                        num_epochs=config_training["epochs"],
                                                        track_metrics=config["track_metrics"],
                                                        )
    

    log_training(
                config=config,
                state=state,
                loss_tracker=loss_tracker,
                accuracy_tracker=accuracy_tracker,
                )
