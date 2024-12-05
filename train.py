import argparse
from jax import devices

from _src.Processors import MLP, CNN
from _src.get_data import get_mnsit_dataloaders
from _src.utils_functions import train_model

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
    parser.add_argument('--features-shapes', type=tuple, default=(32, 64, 256, 10), 
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
    assert len(args.features_shapes) == 4, "model has 4 layers"
    assert args.features_shapes[-1] == 10, "Last output shape must be 10 for MNIST dataset"
    
    # Print configuration for verification
    print("Model Configuration:")
    print(f"Network: {args.model}")
    print(f"Epochs: {args.epochs}")
    if args.model == "CNN": print(f"Kernel size: {args.kernel_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Momentum: {args.momentum}")
    print(f"Out Channels: {args.features_shapes}")
    print(f"Track metrics: {args.track_metrics}")
    
    # List all available devices
    devices = devices()
    print("Available devices:")
    for device in devices:
        print(device)
    
    # Initialize the model
    if args.model == "MLP":
        network = MLP(features_shapes=args.features_shapes)
    elif args.model == "CNN":
        network = CNN(kernel_size=(args.kernel_size,)*2, features_shapes=args.features_shapes)
    
    
    # MNIST dataloaders
    train_ds, test_ds = get_mnsit_dataloaders(batch_size=args.batch_size, model_type=args.model)
    
    
    # Train the model
    state, loss_tracker, accuracy_tracker = train_model(
                                                        model=network,
                                                        train_ds=train_ds,
                                                        test_ds=test_ds,
                                                        lr=args.learning_rate,
                                                        momentum=args.momentum,
                                                        num_epochs=args.epochs,
                                                        track_metrics=args.track_metrics,
                                                        )
    
    args_dict = vars(args)
    if args.model == "MLP": del args_dict["kernel_size"]

    log_training(
                args_dict=args_dict,
                state=state,
                loss_tracker=loss_tracker,
                accuracy_tracker=accuracy_tracker,
                track_metrics=args.track_metrics,
                )
