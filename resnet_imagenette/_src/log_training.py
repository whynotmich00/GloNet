import os
import pickle
import json
import jax.tree_util as jtu
from typing import Dict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [20, 20]


def save_training_files(flags, state, folder_name: str):
    # Save the training state as a pickle file
    state_file = os.path.join(folder_name, "train_state_params.pkl")
    with open(state_file, "wb") as f:
        pickle.dump(state.params, f)
    
    # Save the args dict as a json file
    args_file = os.path.join(folder_name, "config.json")
    with open(args_file, "w") as f:
        json.dump(flags.flag_values_dict(), f)

def create_pdf_report(flags, folder_name: str):
    # Create PDF report
    pdf_file = os.path.join(folder_name, "training_report.pdf")
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    elements.append(Paragraph("Training Report", styles['Heading1']))
    elements.append(Spacer(1, 20))

    # Add configuration parameters
    elements.append(Paragraph("Configuration Parameters:", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    # Convert flags to table data
    flag_data = [[key, str(value)] for key, value in flags.flag_values_dict().items()]
    flag_table = Table([['Parameter', 'Value']] + flag_data)
    flag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(flag_table)
    elements.append(Spacer(1, 20))

    # Add metrics plot
    if os.path.exists(os.path.join(folder_name, "training_metrics.png")):
        elements.append(Paragraph("Training Metrics:", styles['Heading2']))
        elements.append(Spacer(1, 10))
        elements.append(Image(os.path.join(folder_name, "training_metrics.png"), width=450, height=300))

    # Generate PDF
    doc.build(elements)

def plot_training_metrics(metrics: Dict, folder_name: str):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for i, (metric_key, metric_value) in enumerate(metrics.items()):
        if metric_key == "l1_leaf_norm_history":
            # Plot the last step of the history L1 norms of the parameters collected during training
            # This is done to show the usage of the network parameters by the model
            ax = axes[i // 2, i % 2]
            l1_norm = metric_value[-1]  # this is a dictionary
            l1_norm_flat, _ = jtu.tree_flatten_with_path(l1_norm)
            layer_keys = [jtu.keystr(name).replace("']['", " ").replace("['", "").replace("']", "") for name, _ in l1_norm_flat]
            l1_values = [value for _, value in l1_norm_flat]
            
            ax.plot(l1_values, marker="o", label=metric_key, linestyle="None", alpha=0.8)
            ax.set_xticks(range(len(layer_keys)))
            ax.set_xticklabels(layer_keys, rotation=45, ha="right")
            ax.set_ylabel("L1 Norm")
            ax.set_title("L1 Param Norm by Layer")
            ax.grid(True)
            ax.legend()
        
        elif metric_key == "l1_intermediate_output_norm_history":
            # Plot the last step of the history L1 norms of the outputs collected during training
            # This is done to show the usage of the network outputs by the model
            ax = axes[i // 2, i % 2]
            # the mean over the batch output of the last step of the history
            l1_norm = metric_value[-1]
            l1_norm_flat, _ = jtu.tree_flatten_with_path(l1_norm)
            layer_keys = [jtu.keystr(leaf_key).replace("['__call__'][0]", "") for leaf_key, _ in l1_norm_flat]
            layer_keys = [x.replace("['", "").replace("']", "") for x in layer_keys]
            layer_values = [val for _, val in l1_norm_flat]
            ax.plot(layer_values, marker="o", label=metric_key, linestyle="None", alpha=0.8)
            ax.set_xticks(range(len(layer_values)))
            ax.set_xticklabels(layer_keys, rotation=45, ha="right")
            ax.set_ylabel("L1 Norm")
            ax.set_yscale("log")
            ax.set_title("L1 Output Norm by Layer")
            ax.grid(True)
            ax.legend()
            
        else:
            ax = axes[i // 2, i % 2]
            ax.plot(metric_value, label=metric_key, marker="o", alpha=0.8)
            ax.set_ylabel(metric_key)
            ax.set_title(metric_key.replace("_", " ").title())
            ax.grid()
            ax.legend()
        
    fig.tight_layout()
    fig.savefig(os.path.join(folder_name, "training_metrics.png"))
    plt.close()

def log_training(flags, state, metrics: Dict):
    # Create a unique folder for each training session
    parent_folder_name = f"training_logs/{flags.model}"
    os.makedirs(parent_folder_name, exist_ok=True)
    # session folder
    training_id = len(os.listdir(parent_folder_name)) + 1
    folder_name = os.path.join(parent_folder_name, f"session_{training_id}")
    os.makedirs(folder_name, exist_ok=True)

    save_training_files(flags, state, folder_name)
    plot_training_metrics(metrics, folder_name)
    create_pdf_report(flags, folder_name)
    
    print(f"Training log saved in: {folder_name}")