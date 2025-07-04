from typing import Tuple

import torch
from dgl import DGLGraph
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch import Tensor, nn


def evaluating_model(
    model: nn.Module,
    g: DGLGraph,
    labels: Tensor,
    idx: Tensor,
    device: torch.device,
    threshold: float = None
) -> Tuple[Tensor, float, float, float, float, float, Tensor, float]:
    """
    Evaluates the model's performance on the specified subset of nodes.

    Applies the GNN model to the input graph and computes classification metrics
    (accuracy, F1 score, precision, recall, AUC) over the node indices provided.
    Also determines predictions using a dynamic or fixed threshold over the class probabilities.

    Args:
        model (nn.Module): Trained GNN model.
        g (DGLGraph): DGL graph containing node data and structure.
        labels (Tensor): Ground-truth labels for all nodes.
        idx (Tensor): Indices of the nodes to evaluate (e.g., test or validation set).
        device (torch.device): Device to run evaluation on.
        threshold (float, optional): Probability threshold to convert scores into class predictions.
                                     If None, an optimal threshold is determined via F1 score sweep.

    Returns:
        Tuple:
            - logits (Tensor): Raw model outputs for all nodes.
            - acc (float): Accuracy on the specified node indices.
            - f1 (float): F1 score.
            - precision (float): Precision score.
            - recall (float): Recall score.
            - auc_score (float): ROC AUC score.
            - preds (Tensor): Predicted binary labels.
            - threshold (float): Threshold used to binarize probabilities.
    """
    model.eval()
    with torch.no_grad():
        g = g.to(device)
        features = g.ndata['feat'].to(device)
        labels = labels.to(device)
        idx = idx.to(device)

        logits = model(g, features)
        logits = logits.to(device)
        probs = torch.softmax(logits[idx], dim=1)
        y_scores = probs[:, 1].to(device)
        y_true = labels[idx].to(device)

        if threshold is None:
            threshold = evaluate_threshold_sweep(y_true, y_scores)
        else:
            print(f"Using fixed threshold = {threshold:.2f}")

        preds = (y_scores > threshold).long()
        acc = (preds == y_true).float().mean().item()
        f1 = f1_score(y_true, preds, average='binary', zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        auc_score = roc_auc_score(y_true, preds)
        return logits, acc, f1, precision, recall, auc_score, preds, threshold


def evaluate_threshold_sweep(
    y_true: Tensor,
    y_scores: Tensor
) -> float:
    """
    Performs a sweep over thresholds to find the best F1 score while maintaining a minimum recall.

    Args:
        y_true (Tensor): Ground-truth binary labels.
        y_scores (Tensor): Predicted probabilities for the positive class.

    Returns:
        float: Optimal threshold that maximizes F1 score while ensuring recall ≥ 0.7.
    """
    thresholds = [i / 100 for i in range(5, 96, 5)]
    best_f1 = -1.0
    best_threshold = 0.05

    for t in thresholds:
        y_pred = (y_scores > t).long()
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if rec >= 0.7 and f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold
