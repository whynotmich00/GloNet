from jax import devices
from time import time

from _src.train_lib import train_model
from _src.log_training import log_training

from absl import flags, app


FLAGS = flags.FLAGS

flags.DEFINE_string("model", "ResNet34", "Architecture of GloNet or ResNet")
flags.DEFINE_boolean("use_batch_norm", True, "Whether to use batch normalization in the model")
flags.DEFINE_integer("epochs", 10, "Number of training epochs")
flags.DEFINE_float("l2_reg_alpha", 0., "Use L2 regularization ")
flags.DEFINE_integer("batch_size", 64, "Batch size for training")
flags.DEFINE_string("optimizer", "ADAM", "Training optimizer")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for optimization")
flags.DEFINE_float("momentum", 0.0, "Momentum regularization")
flags.DEFINE_integer("features", 128, "Output dimension for each hidden layer in the network")
flags.DEFINE_integer("num_layers", 4, "Number of layers of the network")
flags.DEFINE_string("result_dir", "training_results", "Directory to save training logs and checkpoints")


def main(argv):
    # Print configuration for verification
    print("Model Configuration:")
    print(f"Model: {FLAGS.model}")
    print(f"Epochs: {FLAGS.epochs}")
    print(f"Batch Size: {FLAGS.batch_size}")
    print(f"Optimizer: {FLAGS.optimizer}")
    print(f"Learning Rate: {FLAGS.learning_rate}")
    print(f"Momentum: {FLAGS.momentum}") if FLAGS.optimizer == "SGD" else None
    print(f"Features: {FLAGS.features}")
    print(f"Result directory: {FLAGS.result_dir}")
     
    # Train the model
    start_time = time()
    state, metrics = train_model(flags=FLAGS)
    print(f"Training time: {time() - start_time:.4f} seconds")
    
    # Save the training and report the metrics
    log_training(flags=FLAGS, state=state, metrics=metrics,)
    

if __name__ == "__main__":
    # List all available devices
    devices = devices()
    print("Available devices:")
    for device in devices:
        print(device)
    
    app.run(main)
