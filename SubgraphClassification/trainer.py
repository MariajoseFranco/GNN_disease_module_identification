from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from dgl import DGLGraph
from evaluator import evaluating_model


def training_loop(
    model: nn.Module,
    g: DGLGraph,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epochs: int,
    device: torch.device
) -> Tuple[
    List[float], List[float], List[float], List[float], List[float], float
]:
    """
    Trains a GNN model for node classification over a subgraph.

    This function iterates over a number of epochs, performing training using the provided
    optimizer and loss function. After each epoch, it evaluates the model on the validation set,
    keeping track of the best model based on validation F1 score.

    Args:
        model (nn.Module): The GNN model to train.
        g (DGLGraph): The DGL graph containing node features and structure.
        labels (torch.Tensor): Ground-truth labels for each node.
        train_idx (torch.Tensor): Tensor containing indices of training nodes.
        val_idx (torch.Tensor): Tensor containing indices of validation nodes.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        loss_fn (Callable): Loss function (e.g., CrossEntropyLoss or FocalLoss).
        epochs (int): Number of training epochs.
        device (torch.device): The device (CPU or GPU) on which to run the training.

    Returns:
        Tuple:
            - losses (List[float]): Training loss for each epoch.
            - val_accuracy (List[float]): Validation accuracy per epoch.
            - val_f1s (List[float]): Validation F1 score per epoch.
            - val_precisions (List[float]): Validation precision per epoch.
            - val_recalls (List[float]): Validation recall per epoch.
            - best_threshold (float): Threshold that achieved the best F1 score.
    """
    model.train()

    g = g.to(device)
    features = g.ndata['feat'].to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    losses = []
    val_accuracy = []
    val_f1s = []
    val_precisions = []
    val_recalls = []

    best_f1 = -1
    best_model_state = None
    best_threshold = 0.5

    for epoch in range(epochs):
        logits = model(g, features)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, acc, f1, prec, rec, auc, _, current_thresh = evaluating_model(
            model, g, labels, val_idx, device
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict()
            best_threshold = current_thresh

        val_accuracy.append(acc)
        val_f1s.append(f1)
        val_precisions.append(prec)
        val_recalls.append(rec)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch}, Loss {loss.item():.4f}, Val Acc {acc:.4f}, "
                f"Val F1 {f1:.4f}, Val Prec {prec:.4f}, Val Rec {rec:.4f}, Val AUC {auc:.4f}"
            )

    print(f"Best Threshold: {best_threshold:.2f}")
    model.load_state_dict(best_model_state)
    return losses, val_accuracy, val_f1s, val_precisions, val_recalls, best_threshold
