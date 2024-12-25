import os
import pickle
import json
from jax.tree_util import tree_flatten
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [20, 20]


def log_training(config, state, loss_tracker: None, accuracy_tracker: None):
    # Create a unique folder for each training session
    training_id = sum(f != ".gitignore" for f in os.listdir("training_logs")) if os.path.exists("training_logs") else 0
    folder_name = f"training_logs/{args_dict["model"]}_session_{training_id}"
    os.makedirs(folder_name, exist_ok=True)

    # Save the training state as a pickle file
    state_file = os.path.join(folder_name, "train_state_params.pkl")
    with open(state_file, "wb") as f:
        pickle.dump(state.params, f)
    
    # Save the args dict as a json file
    args_file = os.path.join(folder_name, "config.json")
    with open(args_file, "w") as f:
        json.dump(args_dict, f)

    if loss_tracker is not None:
        # Plot and save the loss values
        loss_training, loss_validation = loss_tracker.metrics["Training"], loss_tracker.metrics["Validation"]
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        # Plot data on each subplot
        ax1.plot(tree_flatten(loss_training)[0], label="Training Loss")
        ax2.plot(tree_flatten(loss_validation)[0], label="Validation Loss")

        # Set axis labels for each subplot
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Loss")
        # Set titles for each subplot
        ax1.set_title("Training")
        ax2.set_title("Validation")
        # Add legends to each subplot
        ax1.legend()
        ax2.legend()
        # Set a main title for the entire figure
        fig.suptitle("Loss over Epochs")

        loss_file = os.path.join(folder_name, "loss_plot.jpeg")
        plt.savefig(loss_file, format="jpeg")
        plt.close()

    if accuracy_tracker is not None:
        # Plot and save the accuracy values
        accuracy_training, accuracy_validation = accuracy_tracker.metrics["Training"], accuracy_tracker.metrics["Validation"]
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        # Plot data on each subplot
        ax1.plot(tree_flatten(accuracy_training)[0], label="Training Accuracy")
        ax2.plot(tree_flatten(accuracy_validation)[0], label="Validation Accuracy")

        # Set axis labels for each subplot
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Accuracy")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Accuracy")
        # Set titles for each subplot
        ax1.set_title("Training")
        ax2.set_title("Validation")
        # Add legends to each subplot
        ax1.legend()
        ax2.legend()
        # Set a main title for the entire figure
        fig.suptitle("Accuracy over Epochs")

        accuracy_file = os.path.join(folder_name, "accuracy_plot.jpeg")
        plt.savefig(accuracy_file, format="jpeg")
        plt.close()

    print(f"Training log saved in: {folder_name}")