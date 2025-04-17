from jax import devices
from flax import nnx

from _src.Processors import CNN, CNN_nnx
from _src.get_data import get_mnsit_dataloaders
from _src.utils_functions import train_model
from _src.config import Config
from _src.log_training import log_training

from absl import flags, app


FLAGS = flags.FLAGS
 
flags.DEFINE_integer("epochs", 10, "Number of training epochs")
flags.DEFINE_integer("batch_size", 256, "Batch size for training")
flags.DEFINE_string("optimizer", "SGD", "Training optimizer")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for optimization")
flags.DEFINE_float("momentum", 0.0, "Momentum regularization")
flags.DEFINE_list("features", [32, 64, 256, 10], "list")
flags.DEFINE_integer("kernel_size", 4, "Convolution kernel size")
flags.DEFINE_string("framework", "plain", "flax linen or nnx?")
flags.DEFINE_boolean("track_metrics", True, "Log losses and accuracy during training and validation")


def main(argv):
    
    assert len(FLAGS.features) == 4, "model has 4 layers"
    assert FLAGS.features[-1] == 10, "Last output shape must be 10 for MNIST dataset"
    
    # Print configuration for verification
    print("Model Configuration:")
    print(f"Epochs: {FLAGS.epochs}")
    print(f"Kernel size: {FLAGS.kernel_size}")
    print(f"Batch Size: {FLAGS.batch_size}")
    print(f"Learning Rate: {FLAGS.learning_rate}")
    print(f"Momentum: {FLAGS.momentum}") if FLAGS.optimizer == "SGD" else None
    print(f"Out Channels: {FLAGS.features}")
    print(f"Track metrics: {FLAGS.track_metrics}")

    # # Print configuration for verification
    # config = Config().as_dict(FLAGS)
    # config_training, config_model, _, config_optimizer = config.items()
    
    # print("Model Configuration:")
    # print(f"Network: {config_model["model"]}")
    # print(f"Epochs: {config_training["epochs"]}")
    # if config_model["model"] == "CNN": print(f"Kernel size: {config_model["kernel_size"]}")
    # print(f"Batch Size: {config_training["batch_size"]}")
    # print(f"Learning Rate: {config_optimizer["learning_rate"]}")
    # print(f"Momentum: {config_optimizer["momentum"]}") if config_optimizer["optimizer"] == "SGD" else None
    # print(f"Out Channels: {config_model["features_shapes"]}")
    # print(f"Track metrics: {config["track_metrics"]}")
    
    rngs = nnx.Rngs(0)
    # Initialize the model
    network = CNN(kernel_size=(FLAGS.kernel_size,)*2, features=FLAGS.features)#if FLAGS.framework == "plain" else CNN_nnx(kernel_size=(FLAGS.kernel_size,)*2, features=FLAGS.features, rngs=rngs)
    
    
    # MNIST dataloaders
    train_ds, test_ds = get_mnsit_dataloaders(batch_size=FLAGS.batch_size)
    
    config_optimizer = {"optimizer": FLAGS.optimizer,
                        "lr": FLAGS.learning_rate,
                        "momentum": FLAGS.momentum,}
    
    # Train the model
    state, loss_tracker, accuracy_tracker = train_model(model=network,
                                                        train_ds=train_ds,
                                                        test_ds=test_ds,
                                                        config_optimizer=config_optimizer,
                                                        flags=FLAGS)
    

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




# if __name__ == "__main__":
    
#     # Parse command-line arguments
#      = parse_arguments()
    
#     assert args.model in ["MLP", "CNN"], "model must be 'MLP' or 'CNN'"
#     assert len(args.features) == 4, "model has 4 layers"
#     assert args.features[-1] == 10, "Last output shape must be 10 for MNIST dataset"
    
#     # Print configuration for verification
#     config = Config().as_dict(args)
#     config_training, config_model, _, config_optimizer = config.items()
    
#     print("Model Configuration:")
#     print(f"Network: {config_model["model"]}")
#     print(f"Epochs: {config_training["epochs"]}")
#     if config_model["model"] == "CNN": print(f"Kernel size: {config_model["kernel_size"]}")
#     print(f"Batch Size: {config_training["batch_size"]}")
#     print(f"Learning Rate: {config_optimizer["learning_rate"]}")
#     print(f"Momentum: {config_optimizer["momentum"]}") if config_optimizer["optimizer"] == "SGD" else None
#     print(f"Out Channels: {config_model["features_shapes"]}")
#     print(f"Track metrics: {config["track_metrics"]}")
    
#     # List all available devices
#     devices = devices()
#     print("Available devices:")
#     for device in devices:
#         print(device)
    
#     # Initialize the model
#     if config["model"] == "MLP":
#         network = MLP(features_shapes=config_model["features_shapes"])
#     elif config["model"] == "CNN":
#         network = CNN(kernel_size=(config_model["kernel_size"],)*2, features_shapes=config_model["features_shapes"])
    
    
#     # MNIST dataloaders
#     train_ds, test_ds = get_mnsit_dataloaders(batch_size=config_training["batch_size"], model_type=config_model["model"])
    
    
#     # Train the model
#     state, loss_tracker, accuracy_tracker = train_model(
#                                                         model=network,
#                                                         train_ds=train_ds,
#                                                         test_ds=test_ds,
#                                                         config_optimizer=config_optimizer,
#                                                         num_epochs=config_training["epochs"],
#                                                         track_metrics=config["track_metrics"],
#                                                         )
    

#     log_training(
#                 config=config,
#                 state=state,
#                 loss_tracker=loss_tracker,
#                 accuracy_tracker=accuracy_tracker,
#                 )
        
