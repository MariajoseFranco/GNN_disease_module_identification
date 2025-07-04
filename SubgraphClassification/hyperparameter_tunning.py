import optuna
import torch
from focal_loss import FocalLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from trainer import training_loop

from SubgraphClassification.GNN_encoder import GNN


def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    This function defines the search space and trains the GNN model
    with the sampled hyperparameters. It returns the best validation F1 score
    for a given trial.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object for suggesting hyperparameters.

    Returns:
        float: The best validation F1 score achieved during training.
    """
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

    _, _, val_f1s, _, _, _ = training_loop(
        model, g, labels, train_idx, val_idx, optimizer, loss_fn, 100, device
    )

    return max(val_f1s)


def run_optuna_tuning(
    g_: torch.Tensor,
    features_: torch.Tensor,
    labels_: torch.Tensor,
    train_idx_: torch.Tensor,
    val_idx_: torch.Tensor,
    disease_name: str,
    path: str
) -> optuna.trial.FrozenTrial:
    """
    Runs Optuna hyperparameter tuning for a specific disease.

    It initializes the study, loads prior trials if available, and optimizes
    the objective function using the provided graph and data.

    Args:
        g_ (torch.Tensor): DGLGraph for the subgraph of the current disease.
        features_ (torch.Tensor): Feature matrix of the graph nodes.
        labels_ (torch.Tensor): Ground-truth node labels.
        train_idx_ (torch.Tensor): Training indices.
        val_idx_ (torch.Tensor): Validation indices.
        disease_name (str): Name of the disease used to name the Optuna study.
        path (str): Path to save the Optuna study database.

    Returns:
        optuna.trial.FrozenTrial: The best trial object containing optimal parameters.
    """
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
