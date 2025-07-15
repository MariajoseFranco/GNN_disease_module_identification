from typing import Callable, Dict, List, Tuple

import torch
from dgl import DGLGraph
from evaluator import evaluating_model
from torch import Tensor
from torch.optim import Optimizer


def training_loop(
    epochs: int,
    model: torch.nn.Module,
    train_pos_g: DGLGraph,
    train_neg_g: DGLGraph,
    train_g: DGLGraph,
    val_pos_g: DGLGraph,
    val_neg_g: DGLGraph,
    features: Dict[str, Tensor],
    optimizer: Optimizer,
    pred: Callable,
    edge_type: Tuple[str, str, str],
    loss_fn: Callable,
    device: torch.device
) -> Tuple[Dict[str, Tensor], List[float], List[float], List[float], List[float], List[float], float]:
    """
    Trains a heterogeneous GNN model for link prediction over multiple epochs,
    using dynamic threshold selection based on validation F1-score.

    Args:
        model (torch.nn.Module): The GNN model.
        train_pos_g (DGLGraph): Graph containing positive training edges.
        train_neg_g (DGLGraph): Graph containing negative training edges.
        train_g (DGLGraph): The full graph (with training edges only) for message passing.
        val_pos_g (DGLGraph): Graph containing positive validation edges.
        val_neg_g (DGLGraph): Graph containing negative validation edges.
        features (Dict[str, Tensor]): Dictionary of input node features keyed by node type.
        optimizer (torch.optim.Optimizer): Optimizer used for model parameter updates.
        pred (Callable): Link prediction module (e.g., DotPredictor or MLPPredictor).
        edge_type (Tuple[str, str, str]): The edge type for link prediction (src, relation, dst).
        loss_fn (Callable): Loss function (e.g., Binary Cross-Entropy or Focal Loss).

    Returns:
        Tuple containing:
            - h (Dict[str, Tensor]): Final node embeddings for each node type.
            - losses (List[float]): List of training losses per epoch.
            - val_accuracys (List[float]): Validation accuracy per epoch.
            - val_f1s (List[float]): Validation F1-score per epoch.
            - val_precs (List[float]): Validation precision per epoch.
            - val_recs (List[float]): Validation recall per epoch.
            - best_threshold (float): Best decision threshold selected on validation data.
    """
    best_f1 = -1
    best_state = None
    best_threshold = 0.5

    losses = []
    val_accuracys = []
    val_f1s = []
    val_recs = []
    val_precs = []
    for epoch in range(epochs):
        model.train()
        h = model(train_g, features)
        pos_score = pred(train_pos_g, h, etype=edge_type, use_seed_score=True)
        neg_score = pred(train_neg_g, h, etype=edge_type, use_seed_score=True)

        scores = torch.cat([pos_score, neg_score]).to(device)
        labels = torch.cat([
            torch.ones(pos_score.shape[0]),
            torch.zeros(neg_score.shape[0])
        ]).float().to(device)

        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        _, _, _, val_prec, val_rec, val_f1, val_auc, val_acc, current_thresh = evaluating_model(
            val_pos_g, val_neg_g, pred, h, edge_type, device
        )

        # Save best model state based on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            best_threshold = current_thresh

        # Logging per epoch
        losses.append(loss.item())
        val_accuracys.append(val_acc)
        val_f1s.append(val_f1)
        val_recs.append(val_rec)
        val_precs.append(val_prec)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | Val Accuracy: {val_acc:.4f} | "
                f"Val AUC: {val_auc:.4f} | Val Precision: {val_prec:.4f} | "
                f"Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}"
            )

    print(f"Best Threshold selected: {best_threshold:.2f}")

    # Restore best model
    model.load_state_dict(best_state)

    return h, losses, val_accuracys, val_f1s, val_precs, val_recs, best_threshold
