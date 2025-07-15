from typing import Callable, Optional, Tuple

import torch
from dgl import DGLGraph
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch import Tensor


def evaluating_model(
    pos_g: DGLGraph,
    neg_g: DGLGraph,
    pred: Callable,
    h: dict,
    edge_type: Tuple[str, str, str],
    device: torch.device,
    threshold: Optional[float] = None
) -> Tuple[Tensor, Tensor, Tensor, float, float, float, float, float, float]:
    """
    Evaluates the trained link prediction model on the test set.

    Args:
        pos_g (DGLGraph): Graph containing positive edges.
        neg_g (DGLGraph): Graph containing negative edges.
        pred (Callable): Link predictor module (DotPredictor or MLPPredictor).
        h (dict): Node embeddings from the trained GNN model, keyed by node type.
        edge_type (Tuple[str, str, str]): The edge type to predict (source, relation, target).
        device (torch.device): The device (CPU/GPU) for computations.
        threshold (Optional[float], optional): Decision threshold for classification.
            If None, it is selected by maximizing F1-score under recall ≥ 0.7. Defaults to None.

    Returns:
        Tuple containing:
            - pos_score (Tensor): Raw model scores for positive test edges.
            - neg_score (Tensor): Raw model scores for negative test edges.
            - labels (Tensor): Ground truth labels (1 for positive, 0 for negative).
            - precision (float): Precision score.
            - recall (float): Recall score.
            - f1 (float): F1 score.
            - auc (float): Area Under the ROC Curve.
            - acc (float): Accuracy.
            - threshold (float): The threshold used for classification.
    """
    with torch.no_grad():
        pos_score = pred(pos_g, h, etype=edge_type, use_seed_score=False)
        neg_score = pred(neg_g, h, etype=edge_type, use_seed_score=False)

        scores = torch.cat([pos_score, neg_score]).to(device)
        probs = torch.sigmoid(scores).to(device)
        labels = torch.cat([
            torch.ones(pos_score.shape[0]),
            torch.zeros(neg_score.shape[0])
        ]).float().to(device)

        if threshold is None:
            threshold = evaluate_threshold_sweep(labels, scores)
        else:
            print(f"Using fixed threshold = {threshold:.2f}")

        preds = (probs > threshold).long()
        acc = (preds == labels).float().mean().item()
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        return pos_score, neg_score, labels, precision, recall, f1, auc, acc, threshold


def evaluate_threshold_sweep(
    y_true: Tensor,
    y_scores: Tensor
) -> float:
    """
    Performs a sweep over thresholds to find the best F1-score under a recall constraint.

    Args:
        y_true (Tensor): Ground truth labels (1 for positive, 0 for negative).
        y_scores (Tensor): Raw scores (logits) from the model.

    Returns:
        float: The threshold that maximizes F1-score while ensuring recall ≥ 0.7.
    """
    thresholds = [i / 100 for i in range(5, 96, 5)]
    best_f1 = -1
    best_threshold = 0.05

    for t in thresholds:
        y_pred = (y_scores > t).long()
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if rec >= 0.7 and f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold
