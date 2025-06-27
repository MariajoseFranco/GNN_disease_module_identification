import optuna
import torch
from focal_loss import FocalLoss
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from SubgraphClassification.GNN_sage import GNN


def objective(trial):
    hidden_feats = trial.suggest_categorical("hidden_feats", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    layer_type = trial.suggest_categorical("layer_type", ["GraphConv", "SAGEConv"])
    proportion = trial.suggest_float("proportion", 0.4, 1.0)

    model = GNN(
        in_feats=features.shape[1],
        hidden_feats=hidden_feats,
        num_layers=num_layers,
        layer_type=layer_type,
        dropout=dropout
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_focal_loss:
        loss_fn = FocalLoss()
    else:
        weights = torch.tensor([1.0, (num_0 / num_1) * proportion], dtype=torch.float32).to(device)
        loss_fn = CrossEntropyLoss(weight=weights)

    losses, val_acc, val_f1s, _, _, best_threshold = training_loop(
        model, g, labels, train_idx, val_idx, optimizer, loss_fn
    )

    return max(val_f1s)


def run_optuna_tuning(g_, features_, labels_, train_idx_, val_idx_, disease_name, path):
    db_path = f"{path}/optuna_study.db"
    storage = f"sqlite:///{db_path}"
    study_name = f"{disease_name}_study"

    global g, features, labels, train_idx, val_idx, num_0, num_1, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = g_.to(device)
    features = features_.to(device)
    labels = labels_.to(device)
    train_idx = train_idx_
    val_idx = val_idx_
    num_0 = (labels[train_idx] == 0).sum().item()
    num_1 = (labels[train_idx] == 1).sum().item()

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True
    )
    print(f"Loaded study with {len(study.trials)} trials.")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print(study.best_trial)
    return study.best_trial


def training_loop(model, g, labels, train_idx, val_idx, optimizer, loss_fn):
    """
    Trains the GNN model for link prediction using positive and negative edges.

    Args:
        model (nn.Module): The GNN model.
        predictor (nn.Module): The predictor module that computes edge scores.
        train_pos_u (Tensor): Source nodes of positive training edges.
        train_pos_v (Tensor): Destination nodes of positive training edges.
        train_neg_u (Tensor): Source nodes of negative training edges.
        train_neg_v (Tensor): Destination nodes of negative training edges.
        g (DGLGraph): The homogeneous protein-protein interaction graph.
        features (Tensor): Node feature matrix.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (torch.nn.Module): Loss function (e.g., BCEWithLogitsLoss).

    Returns:
        None: Trains the model in-place.
    """
    epochs = 100
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
    best_threshold = 0.5  # Default fallback

    for epoch in range(epochs):
        logits = model(g, features)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc, f1, prec, rec, _, current_thresh = evaluating_model(
            model, g, labels, val_idx
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
                f"Epoch {epoch}, Loss {loss.item():.4f}"
                f", Val Acc {acc:.4f}, Val F1 {f1:.4f}"
                f", Val Prec {prec:.4f}, Val Rec {rec:.4f}"
            )
    print(f"Best Threshold: {best_threshold:.2f}")
    model.load_state_dict(best_model_state)
    return losses, val_accuracy, val_f1s, val_precisions, val_recalls, best_threshold


def evaluating_model(model, g, labels, idx, threshold=None):
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
        return acc, f1, precision, recall, preds, threshold


def evaluate_threshold_sweep(y_true, y_scores):
    thresholds = [i / 100 for i in range(5, 96, 5)]  # from 0.05 to 0.95
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
