import argparse
from jax import devices

from _src.Processors import MLP
from _src.get_data import get_mnsit_dataloaders
from _src.utils_functions import train_model
from _src.config import Config
from _src.log_training import log_training

from absl import flags, app

def parse_arguments():
    """
    Parse command-line arguments for model configuration.
    """
    parser = argparse.ArgumentParser(description="Configurable Machine Learning Model")
    
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




FLAGS = flags.FLAGS
 
flags.DEFINE_integer("epochs", 10, "Number of training epochs")
flags.DEFINE_integer("batch_size", 256, "Batch size for training")
flags.DEFINE_string("optimizer", "SGD", "Training optimizer")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for optimization")
flags.DEFINE_float("momentum", 0.0, "Momentum regularization")
flags.DEFINE_list("features", [32, 64, 256, 10], "list")
flags.DEFINE_integer("kernel_size", 4, "Convolution kernel size")
flags.DEFINE_boolean("track_metrics", True, "Log losses and accuracy during training and validation")


def main(argv):
    
    assert len(FLAGS.features) == 4, "model has 4 layers"
    assert FLAGS.features[-1] == 10, "Last output shape must be 10 for MNIST dataset"
    
    # Print configuration for verification
    print("Model Configuration:")
    print(f"Epochs: {FLAGS.epochs}")
    print(f"Batch Size: {FLAGS.batch_size}")
    print(f"Learning Rate: {FLAGS.learning_rate}")
    print(f"Momentum: {FLAGS.momentum}") if FLAGS.optimizer == "SGD" else None
    print(f"Features: {FLAGS.features}")
    print(f"Track metrics: {FLAGS.track_metrics}")
    
    # Initialize the model
    network = MLP(features_shapes=FLAGS.features)
    
    # MNIST dataloaders
    train_ds, test_ds = get_mnsit_dataloaders(batch_size=FLAGS.batch_size)
    
    config_optimizer = {"optimizer": FLAGS.optimizer,
                        "lr": FLAGS.learning_rate,
                        "momentum": FLAGS.momentum,}
    
    # Train the model
    state, loss_tracker, accuracy_tracker = train_model(model=network,
                                                        train_ds=train_ds,
                                                        test_ds=test_ds,
                                                        flags=FLAGS,
                                                        config_optimizer=config_optimizer)
    

    log_training(flags=FLAGS,
                state=state,
                loss_tracker=loss_tracker,
                accuracy_tracker=accuracy_tracker)
    

if __name__ == "__main__":
    # List all available devices
    devices = devices()
    print("Available devices:")
    for device in devices:
        print(device)
    
    
    app.run(main)
