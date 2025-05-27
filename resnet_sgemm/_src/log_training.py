import os
import pickle
import json
import jax.tree_util as jtu
import numpy as np
from typing import Dict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from _src.utils import compute_l1_validation_mean_inter_outputs, compute_l1_norms_leaves


# plt.rcParams['figure.figsize'] = [20, 20]


def save_training_files(flags, state, metrics, folder_name: str):
    # Save the training state as a pickle file
    state_file = os.path.join(folder_name, "train_state_params.pkl")
    with open(state_file, "wb") as f:
        params = jtu.tree_map(np.array, state.params)
        pickle.dump(params, f)
    
    # Save the args dict as a json file
    args_file = os.path.join(folder_name, "config.json")
    with open(args_file, "w") as f:
        json.dump(flags.flag_values_dict(), f)
    
    # Save the training metrics as a pickle file
    metrics_file = os.path.join(folder_name, "training_metrics.pkl")
    with open(metrics_file, "wb") as f:
        metrics_memory = jtu.tree_map(np.array, metrics)
        pickle.dump(metrics_memory, f)

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
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Create subplots
    fig, _ = plt.subplots(figsize=(12, 8))
    # Hide the figure-level axes
    fig.axes[0].set_visible(False)
    # Create GridSpec
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    metric_axes = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])]
    param_norm_axes = plt.subplot(gs[1, :])
    interm_outputs_norm_axes = plt.subplot(gs[2, :])
    
    for i, (metric_key, metric_value) in enumerate(metrics.items()):
        if metric_key == "l1_leaf_norm_history":
            print("Plotting L1 norms of the parameters collected during training")
            # Plot the last step of the history L1 norms of the parameters collected during training
            # This is done to show the usage of the network parameters by the model
            ax = param_norm_axes
            l1_norm_leaves = metric_value[0][-1]                  # this is a dictionary
            ordered_keys = metric_value[1]["ordered_keys"]        # this is a list
            ordered_values_by_layer = compute_l1_norms_leaves(l1_norm_leaves, ordered_keys=ordered_keys)
            
            # l1_norm_flat_leaves = jtu.tree_leaves(l1_norm_leaves, is_leaf=lambda x: "bias" in x)
            # l1_norm_by_layer = [l["bias"] + l["kernel"] if "kernel" in l else None for l in l1_norm_flat_leaves]
            # l1_norm_by_layer = list(filter(lambda x: x is not None, l1_norm_by_layer))
            # l1_norm_by_bn_layer = [l["bias"] + l["scale"] if "scale" in l else None for l in l1_norm_flat_leaves]
            # l1_norm_by_bn_layer = list(filter(lambda x: x is not None, l1_norm_by_bn_layer))
            
            ax.plot(ordered_values_by_layer, marker="o", label=metric_key, linestyle="None", alpha=0.8)
            ax.set_ylabel("L1 Norm")
            ax.set_title("L1 Param Norm by Layer")
            ax.grid(True)
            ax.legend()
        
        elif metric_key == "l1_intermediate_output_validation":
            # Plot the last step of the history L1 norms of the outputs collected during training
            # This is done to show the usage of the network outputs by the model
            ax = interm_outputs_norm_axes
            # the mean of l1 norms of the intermediate outputs of the last epoch on validation set
            l1_norms_with_path = jtu.tree_leaves_with_path(metric_value)
            l1_norms = [v for _, v in l1_norms_with_path]
            l1_layers = [k[0].key if "resnet" not in k[0].key else k[0].key + k[1].key for k, _ in l1_norms_with_path]
            ax.plot(l1_norms, marker="o", label=metric_key, linestyle="None", alpha=0.8)
            ax.set_xticks(range(len(l1_layers)))
            ax.set_xticklabels(l1_layers, rotation=45, ha='right')
            ax.set_xlabel("Layers")
            ax.set_ylabel("L1 Norm")
            ax.set_title("Mean of L1 Output Norm by Layer for the Validation Set (last epoch)")
            ax.grid(True)
            ax.legend()
            
        else:
            ax = metric_axes[i % 2]
            ax.plot(metric_value, label=metric_key, marker="o", alpha=0.8)
            ax.set_ylabel(metric_key)
            ax.set_title(metric_key.replace("_", " ").title())
            ax.set_xlabel("Epochs" if "val" in metric_key else "Steps")
            ax.set_yscale("log")
            ax.grid()
            ax.legend()
        
    fig.tight_layout()
    fig.savefig(os.path.join(folder_name, "training_metrics.png"))
    plt.close()

def log_training(flags, state, metrics: Dict):
    # Create a unique folder for each training session
    parent_folder_name = f"{flags.result_dir}/{flags.model}"
    os.makedirs(parent_folder_name, exist_ok=True)
    # session folder
    training_id = len(os.listdir(parent_folder_name)) + 1
    folder_name = os.path.join(parent_folder_name, f"session_{training_id}")
    os.makedirs(folder_name, exist_ok=True)

    save_training_files(flags, state, metrics, folder_name)
    plot_training_metrics(metrics, folder_name)
    create_pdf_report(flags, folder_name)
    
    print(f"Training log saved in: {folder_name}")