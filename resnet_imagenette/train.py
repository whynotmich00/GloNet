from jax import devices
from time import time

from _src.train_lib import train_model
from _src.log_training import log_training

from absl import flags, app


FLAGS = flags.FLAGS

flags.DEFINE_string("model", "ResNet50", "Architecture of GloNet or ResNet") 
flags.DEFINE_integer("epochs", 10, "Number of training epochs")
flags.DEFINE_integer("batch_size", 32, "Batch size for training")
flags.DEFINE_string("optimizer", "SGD", "Training optimizer")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimization")
flags.DEFINE_float("momentum", 0.0, "Momentum regularization")
flags.DEFINE_integer("hidden_dim", 256, "Output dimension for each hidden layer in the network")
flags.DEFINE_integer("num_layers", 4, "Number of layers of the network")


def main(argv):
    # Print configuration for verification
    print("Model Configuration:")
    print(f"Model: {FLAGS.model}")
    print(f"Epochs: {FLAGS.epochs}")
    print(f"Batch Size: {FLAGS.batch_size}")
    print(f"Learning Rate: {FLAGS.learning_rate}")
    print(f"Momentum: {FLAGS.momentum}") if FLAGS.optimizer == "SGD" else None
    print(f"Features: {FLAGS.hidden_dim}")
    
    # Train the model
    start_time = time()
    state, metrics = train_model(flags=FLAGS)
    print(f"Training time: {time() - start_time} seconds")
    
    # Save the training and report the metrics
    log_training(flags=FLAGS, state=state, metrics=metrics,)
    

if __name__ == "__main__":
    # List all available devices
    devices = devices()
    print("Available devices:")
    for device in devices:
        print(device)
    
    app.run(main)
