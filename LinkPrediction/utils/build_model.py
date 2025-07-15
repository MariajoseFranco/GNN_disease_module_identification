from typing import Callable, Dict, Tuple

import dgl
import torch
import torch.nn.functional as F
from focal_loss import FocalLoss
from mlp_predictor import MLPPredictor
from torch import nn

from LinkPrediction.dot_predictor import DotPredictor
from LinkPrediction.heteroGNN import HeteroGNN as GNN


def builder(
    best_params: Dict,
    features: Dict[str, torch.Tensor],
    edge_type: Tuple[str, str, str],
    etypes: list,
    train_pos_g: dgl.DGLHeteroGraph,
    train_neg_g: dgl.DGLHeteroGraph,
    device: torch.device
) -> Tuple[nn.Module, torch.optim.Optimizer, Callable, nn.Module]:
    """
    Constructs the model, optimizer, loss function, and predictor based on the best hyperparameters.

    Args:
        best_params (dict): Best hyperparameters from Optuna.
        features (dict): Node features by node type.
        edge_type (tuple): Edge type for link prediction.
        etypes (list): All edge types in the heterogeneous graph.
        train_pos_g (dgl.DGLHeteroGraph): Graph of positive training edges.
        train_neg_g (dgl.DGLHeteroGraph): Graph of negative training edges.
        device (torch.device): Computation device.

    Returns:
        tuple: (model, optimizer, loss_fn, predictor)
    """
    in_feats = features['disease'].shape[1]
    model = building_model(best_params, in_feats, etypes, features, device)
    optimizer = building_optimizer(best_params, model)
    loss_fn = building_loss(best_params, train_pos_g, train_neg_g, edge_type, device)
    predictor = building_predictor(best_params, in_feats, device)
    return model, optimizer, loss_fn, predictor


def building_model(
    best_params: Dict,
    in_feats: int,
    etypes: list,
    features: Dict[str, torch.Tensor],
    device: torch.device
) -> nn.Module:
    """
    Instantiates the heterogeneous GNN model with the specified hyperparameters.

    Args:
        best_params (dict): Best hyperparameters from Optuna.
        in_feats (int): Input feature size.
        etypes (list): List of edge types.
        features (dict): Node features.
        device (torch.device): Computation device.

    Returns:
        nn.Module: The initialized GNN model.
    """
    model = GNN(
        in_feats=in_feats,
        hidden_feats=best_params["hidden_feats"],
        etypes=etypes,
        node_types=list(features.keys()),
        num_layers=best_params["num_layers"],
        layer_type=best_params["layer_type"],
        aggregator_type=best_params["aggregator_type"],
        dropout=best_params["dropout"]
    ).to(device)
    return model


def building_optimizer(
    best_params: dict,
    model: nn.Module
) -> torch.optim.Optimizer:
    """
    Builds and returns an Adam optimizer configured with the provided parameters.

    Args:
        best_params (dict): Dictionary containing 'lr' and 'weight_decay'.
        model (nn.Module): The model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: Configured Adam optimizer.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    return optimizer


def building_loss(
    best_params: Dict,
    train_pos_g: dgl.DGLHeteroGraph,
    train_neg_g: dgl.DGLHeteroGraph,
    edge_type: Tuple[str, str, str],
    device: torch.device
) -> Callable:
    """
    Selects the loss function (Focal or BCE) based on hyperparameters.

    Args:
        best_params (dict): Best hyperparameters from Optuna.
        train_pos_g (dgl.DGLHeteroGraph): Graph of positive training edges.
        train_neg_g (dgl.DGLHeteroGraph): Graph of negative training edges.
        edge_type (tuple): Edge type for link prediction.
        device (torch.device): Computation device.

    Returns:
        Callable: The loss function.
    """
    if best_params["use_focal"]:
        loss_fn = FocalLoss().to(device)
    else:
        loss_fn = bce_loss_fn(train_pos_g, train_neg_g, edge_type, device)
    return loss_fn


def bce_loss_fn(
    pos_graph: dgl.DGLHeteroGraph,
    neg_graph: dgl.DGLHeteroGraph,
    etype: Tuple[str, str, str],
    device: torch.device
) -> Callable:
    """
    Returns a weighted binary cross-entropy loss function to handle class imbalance.

    Args:
        pos_graph (dgl.DGLHeteroGraph): Positive samples graph.
        neg_graph (dgl.DGLHeteroGraph): Negative samples graph.
        etype (tuple): Edge type to compute loss over.
        device (torch.device): Computation device.

    Returns:
        Callable: Weighted BCE loss function.
    """
    num_pos = pos_graph.num_edges(etype=etype)
    num_neg = neg_graph.num_edges(etype=etype)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    def loss_fn(score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(score, label, pos_weight=pos_weight)

    return loss_fn


def building_predictor(
    best_params: Dict,
    in_feats: int,
    device: torch.device
) -> nn.Module:
    """
    Creates the link predictor module (dot product or MLP).

    Args:
        best_params (dict): Best hyperparameters from Optuna.
        in_feats (int): Input feature size of the node embeddings.
        device (torch.device): Computation device.

    Returns:
        nn.Module: Predictor module for link prediction.
    """
    if best_params["predictor_type"] == "dot":
        pred = DotPredictor().to(device)
    else:
        pred = MLPPredictor(
            in_feats=in_feats,
            hidden_feats=best_params["hidden_feats"]
        ).to(device)
    return pred
