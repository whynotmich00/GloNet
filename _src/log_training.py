import os
import pickle
import json
from jax.tree_util import tree_flatten
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 15]


def log_training(args_dict, state, loss_tracker, accuracy_tracker, track_metrics=True, track_grad_and_params_norms=False):
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

    if track_metrics:
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
    
    if track_grad_and_params_norms:
        
        # Save the gradient norms as a pickle file
        grad_norm_file = os.path.join(folder_name, "grad_norm_history.pkl")
        with open(grad_norm_file, "wb") as f:
            pickle.dump(state.grad_norm_history, f)
        
        plt.figure()
        plt.plot(state.grad_norm_history)
        plt.yscale('log')
        plt.xlabel("Steps")
        plt.ylabel("(log) Norms")
        plt.title("Gradient norms over steps")
        g_plot_file = os.path.join(folder_name, "grad_norms.jpeg")
        plt.savefig(g_plot_file, format="jpeg")
        plt.close()
        
        # Save the gradient norms as a pickle file
        params_norm_file = os.path.join(folder_name, "params_norm_history.pkl")
        with open(params_norm_file, "wb") as f:
            pickle.dump(state.params_norm_history, f)
        
        plt.figure()
        plt.plot(state.params_norm_history)
        plt.yscale('log')
        plt.xlabel("Steps")
        plt.ylabel("(log) Norms")
        plt.title("Params norms over steps")
        p_plot_file = os.path.join(folder_name, "params_norms.jpeg")
        plt.savefig(p_plot_file, format="jpeg")
        plt.close()


    print(f"Training log saved in: {folder_name}")