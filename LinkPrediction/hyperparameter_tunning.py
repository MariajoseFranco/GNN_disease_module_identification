import itertools

import optuna
import torch
import torch.nn.functional as F
from dot_predictor import DotPredictor
from focal_loss import FocalLoss
from mlp_predictor import MLPPredictor
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.optim import Adam

from LinkPrediction.heteroGNN import HeteroGNN as GNN


def run_optuna_tuning(
    train_pos_g_, train_neg_g_, train_g_,
    val_pos_g_, val_neg_g_,
    edge_type_, features_, path, all_etypes, drug=False
):
    db_path = f"{path}/drugs_optuna_study.db"
    storage = f"sqlite:///{db_path}"
    study_name = "link_prediction_drugs_study"

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move graphs to device
    train_pos_g = train_pos_g_.to(device)
    train_neg_g = train_neg_g_.to(device)
    train_g = train_g_.to(device)
    val_pos_g = val_pos_g_.to(device)
    val_neg_g = val_neg_g_.to(device)

    # Prepare features dictionary
    if drug:
        features = {
            'disease': features_['disease'].to(device),
            'protein': features_['protein'].to(device),
            'drug': features_['drug'].to(device),
            'pathway': features_['pathway'].to(device),
            'phenotype': features_['phenotype'].to(device)
        }
    else:
        features = {
            'disease': features_['disease'].to(device),
            'protein': features_['protein'].to(device)
        }

    # Pass values via closure/global
    def objective(trial):
        hidden_feats = trial.suggest_categorical("hidden_feats", [32, 64, 128, 256])
        aggregator_type = trial.suggest_categorical("aggregator_type", ["mean", "lstm"])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        layer_type = trial.suggest_categorical("layer_type", ["GraphConv", "SAGEConv"])
        predictor_type = trial.suggest_categorical("predictor_type", ["dot", "mlp"])
        use_focal = trial.suggest_categorical("use_focal", [True, False])

        model = GNN(
            in_feats=features['disease'].shape[1],
            hidden_feats=hidden_feats,
            etypes=list(train_g.etypes),
            node_types=list(features.keys()),  # <- ¡clave para que use 'disease', 'protein', 'drug', 'pathway'!
            num_layers=num_layers,
            layer_type=layer_type,
            aggregator_type=aggregator_type,
            dropout=dropout
        ).to(device)

        if predictor_type == "dot":
            pred = DotPredictor().to(device)
        else:
            pred = MLPPredictor(
                in_feats=features['disease'].shape[1],
                hidden_feats=hidden_feats
            ).to(device)

        if use_focal:
            loss_fn = FocalLoss().to(device)
        else:
            loss_fn = bce_loss_fn(train_pos_g, train_neg_g, edge_type_)

        h, _, val_acc, val_f1, _, _, _ = training_loop(
            model, train_pos_g, train_neg_g, train_g,
            val_pos_g, val_neg_g, features,
            optimizer=Adam(
                itertools.chain(model.parameters(), pred.parameters()),
                lr=lr,
                weight_decay=weight_decay
            ),
            pred=pred,
            edge_type=edge_type_,
            loss_fn=loss_fn
        )
        return val_f1

    # Run Optuna
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


def training_loop(
        model,
        train_pos_g,
        train_neg_g,
        train_g,
        val_pos_g,
        val_neg_g,
        features,
        optimizer,
        pred,
        edge_type,
        loss_fn
):
    """
    Trains the GNN model on a heterogeneous graph for link prediction.

    Args:
        model (nn.Module): The GNN model.
        train_pos_g (DGLGraph): Graph containing positive training edges.
        train_neg_g (DGLGraph): Graph containing negative training edges.
        train_g (DGLGraph): The full graph (minus test edges) for message passing.
        features (dict): Node feature dictionary by node type.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        pred (DotPredictor): Predictor module to compute edge scores.
        edge_type (tuple): The edge type to predict (source, relation, target).

    Returns:
        dict: Node embeddings after training.
    """
    best_f1 = -1
    best_state = None
    best_threshold = 0.5
    epochs = 100
    for epoch in range(epochs):
        # forward
        h = model(train_g, features)
        pos_score = pred(train_pos_g, h, etype=edge_type, use_seed_score=True)
        neg_score = pred(train_neg_g, h, etype=edge_type, use_seed_score=True)

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).float().to(device)
        loss = loss_fn(scores, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            _, _, _, val_prec, val_rec, val_f1, val_auc, val_acc, current_thresh = evaluating_model(
                val_pos_g, val_neg_g, pred, h, edge_type
            )
            print(
                f"Epoch {epoch} | Loss: {loss:.4f} | Val Accuracy: {val_acc:.4f} | "
                f"Val AUC: {val_auc:.4f} | Val Precision: {val_prec:.4f} | "
                f"Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = model.state_dict()
                best_threshold = current_thresh
    print(f"Best Threshold: {best_threshold:.2f}")
    model.load_state_dict(best_state)  # Restore best model
    return h, loss, val_acc, val_f1, val_prec, val_rec, best_threshold


def evaluating_model(test_pos_g, test_neg_g, pred, h, edge_type, threshold=None):
    """
    Evaluates the trained model using AUC on the test set.

    Args:
        test_pos_g (DGLGraph): Graph with positive test edges.
        test_neg_g (DGLGraph): Graph with negative test edges.
        pred (DotPredictor): Predictor module to compute edge scores.
        h (dict): Node embeddings from the trained model.
        edge_type (tuple): The edge type for prediction.

    Returns:
        tuple: (pos_score, neg_score, labels)
            - pos_score (Tensor): Scores for positive test edges.
            - neg_score (Tensor): Scores for negative test edges.
            - labels (ndarray): Ground truth labels (1 for positive, 0 for negative).
    """
    with torch.no_grad():
        pos_score = pred(test_pos_g, h, etype=edge_type, use_seed_score=False)
        neg_score = pred(test_neg_g, h, etype=edge_type, use_seed_score=False)

        scores = torch.cat([pos_score, neg_score])
        probs = torch.sigmoid(scores)
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


def bce_loss_fn(pos_graph, neg_graph, etype):
    num_pos = pos_graph.num_edges(etype=etype)
    num_neg = neg_graph.num_edges(etype=etype)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    def loss_fn(score, label):
        return F.binary_cross_entropy_with_logits(score, label, pos_weight=pos_weight)

    return loss_fn
