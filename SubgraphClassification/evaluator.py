import torch
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)


def evaluating_model(model, g, labels, idx, device, threshold=None):
    """
    Evaluates the trained model on test edges and computes accuracy.

    Args:
        model (nn.Module): The GNN model.
        predictor (nn.Module): The predictor module for edge scores.
        test_pos_u (Tensor): Source nodes of positive test edges.
        test_pos_v (Tensor): Destination nodes of positive test edges.
        test_neg_u (Tensor): Source nodes of negative test edges.
        test_neg_v (Tensor): Destination nodes of negative test edges.
        g (DGLGraph): The homogeneous graph.
        features (Tensor): Node feature matrix.

    Returns:
        tuple: (predictions, u_test, v_test)
            - predictions (Tensor): Binary predictions for test edges.
            - u_test (Tensor): Source nodes for test edges.
            - v_test (Tensor): Destination nodes for test edges.
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


def evaluate_threshold_sweep(y_true, y_scores):
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
