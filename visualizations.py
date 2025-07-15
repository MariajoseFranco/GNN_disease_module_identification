from typing import List, Optional, Tuple

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Patch
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve)


def plot_loss_and_metrics(
    losses: List[float],
    val_f1s: List[float],
    val_recalls: List[float],
    val_precisions: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training loss and validation metrics over epochs.

    Args:
        losses (List[float]): Training loss values per epoch.
        val_f1s (List[float]): Validation F1-scores per epoch.
        val_recalls (List[float]): Validation recall values per epoch.
        val_precisions (List[float]): Validation precision values per epoch.
        save_path (Optional[str]): If provided, saves the plot to the given path.

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Train Loss')
    plt.plot(val_f1s, label='Val F1-score')
    plt.plot(val_recalls, label='Val Recall')
    plt.plot(val_precisions, label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Training Progress')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the confusion matrix as a heatmap.

    Args:
        y_true (torch.Tensor): Ground truth binary labels.
        y_pred (torch.Tensor): Predicted binary labels.
        save_path (Optional[str]): If provided, saves the plot to the given path.

    Returns:
        None
    """
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    save_path: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot the Precision-Recall curve.

    Args:
        y_true (torch.Tensor): Ground truth binary labels.
        y_scores (torch.Tensor): Predicted probabilities or scores.
        save_path (Optional[str]): If provided, saves the plot.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Precision and recall arrays.
    """
    plt.figure(figsize=(6, 5))
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return precision, recall


def plot_roc_curve(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    save_path: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot the ROC curve and compute the AUC.

    Args:
        y_true (torch.Tensor): Ground truth binary labels.
        y_scores (torch.Tensor): Predicted probabilities or scores.
        save_path (Optional[str]): If provided, saves the plot.

    Returns:
        Tuple[np.ndarray, np.ndarray]: False positive rate (FPR) and true positive rate (TPR).
    """
    fpr, tpr, _ = roc_curve(y_true.cpu(), y_scores.cpu())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fpr, tpr


def plot_all_curves(
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    curve_type: str = "pr",
    save_path: str = "all_curves.png"
) -> None:
    """
    Plot multiple PR or ROC curves for comparison.

    Args:
        curves (List[Tuple[str, np.ndarray, np.ndarray]]): Each element is a tuple (label, x, y),
            where x and y are arrays of curve points.
        curve_type (str): Either 'pr' (Precision-Recall) or 'roc' (ROC). Defaults to 'pr'.
        save_path (str): Path to save the plot.

    Returns:
        None
    """
    plt.figure()
    for disease, x, y in curves:
        label = f"{disease}"
        plt.plot(x, y, label=label)

    if curve_type == "pr":
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
    else:
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("ROC Curves")

    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_disease_protein_associations(
    g: dgl.DGLHeteroGraph,
    diseases: List[int],
    max_edges: int = 200,
    output_path: Optional[str] = None
) -> None:
    """
    Visualizes disease–protein associations from a heterogeneous DGL graph
    using NetworkX and Matplotlib.

    This function selects edges of type ('disease', 'associates', 'protein')
    for the specified disease nodes
    and visualizes the bipartite graph. A maximum of `max_edges` edges are sampled for clarity.

    Args:
        g (dgl.DGLHeteroGraph): The heterogeneous graph containing disease–protein associations.
            Expected to have an edge type ('disease', 'associates', 'protein').
        diseases (List[int]): List of disease node indices to include in the visualization.
        max_edges (int, optional): Maximum number of edges to visualize. Defaults to 200.
        output_path (Optional[str], optional): If provided, saves the visualization
        to the given path.
        If None, the plot is displayed interactively.

    Returns:
        None
    """
    etype = ('disease', 'associates', 'protein')
    src, dst = g.edges(etype=etype)

    # Filter edges involving the selected diseases
    mask = torch.isin(src, torch.tensor(diseases))
    src = src[mask]
    dst = dst[mask]

    # Limit the number of edges to plot
    if len(src) > max_edges:
        indices = torch.randperm(len(src))[:max_edges]
        src = src[indices]
        dst = dst[indices]

    # Build NetworkX bipartite graph
    G_nx = nx.DiGraph()
    for s, d in zip(src.tolist(), dst.tolist()):
        disease_label = f"disease_{s}"
        protein_label = f"protein_{d}"
        G_nx.add_node(disease_label, bipartite=0)
        G_nx.add_node(protein_label, bipartite=1)
        G_nx.add_edge(disease_label, protein_label)

    # Layout and drawing
    pos = nx.spring_layout(G_nx, k=0.5, seed=42)
    plt.figure(figsize=(16, 10))
    node_colors = ['lightcoral' if n.startswith('disease') else 'skyblue' for n in G_nx.nodes()]
    nx.draw(
        G_nx, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=300,
        font_size=7,
        edge_color='gray',
        alpha=0.9
    )

    # Add legend and title
    legend_elements = [
        Patch(facecolor='lightcoral', label='Diseases'),
        Patch(facecolor='skyblue', label='Proteins')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Disease–Protein Associations")
    plt.axis('off')
    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()
