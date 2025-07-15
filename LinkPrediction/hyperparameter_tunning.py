import itertools

import dgl
import optuna
import torch
from dot_predictor import DotPredictor
from focal_loss import FocalLoss
from mlp_predictor import MLPPredictor
from torch.optim import Adam
from trainer import training_loop

from LinkPrediction.heteroGNN import HeteroGNN as GNN
from LinkPrediction.utils.build_model import bce_loss_fn


def run_optuna_tuning(
    train_pos_g_: dgl.DGLHeteroGraph,
    train_neg_g_: dgl.DGLHeteroGraph,
    train_g_: dgl.DGLHeteroGraph,
    val_pos_g_: dgl.DGLHeteroGraph,
    val_neg_g_: dgl.DGLHeteroGraph,
    edge_type_: tuple,
    features_: dict,
    path: str,
    all_etypes: list,
    drug: bool = False
) -> optuna.trial.FrozenTrial:
    """
    Runs Optuna hyperparameter tuning for a heterogeneous graph link prediction model.

    This function searches for the best combination of GNN architecture parameters
    (hidden units, aggregator, layers, etc.) and training hyperparameters
    (learning rate, dropout, etc.) to maximize validation F1-score.

    Args:
        train_pos_g_ (dgl.DGLHeteroGraph): Positive training graph.
        train_neg_g_ (dgl.DGLHeteroGraph): Negative training graph.
        train_g_ (dgl.DGLHeteroGraph): Full graph for message passing (train edges only).
        val_pos_g_ (dgl.DGLHeteroGraph): Positive validation graph.
        val_neg_g_ (dgl.DGLHeteroGraph): Negative validation graph.
        edge_type_ (tuple): Canonical edge type (src_type, relation, dst_type) to predict.
        features_ (dict): Dictionary of node features by node type.
        path (str): Directory where the Optuna study will be saved.
        all_etypes (list): List of all edge types in the graph.
        drug (bool, optional): Whether the graph includes drug nodes. Defaults to False.

    Returns:
        optuna.trial.FrozenTrial: The best trial found by Optuna, containing
        hyperparameters and performance.
    """
    db_path = f"{path}/optuna_study.db"
    storage = f"sqlite:///{db_path}"
    study_name = "link_prediction_study"

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
            node_types=list(features.keys()),
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
            100, model, train_pos_g, train_neg_g, train_g,
            val_pos_g, val_neg_g, features,
            optimizer=Adam(
                itertools.chain(model.parameters(), pred.parameters()),
                lr=lr,
                weight_decay=weight_decay
            ),
            pred=pred,
            edge_type=edge_type_,
            loss_fn=loss_fn,
            device=device
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
