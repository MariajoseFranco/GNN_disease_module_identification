from typing import Tuple

import torch
from dgl import DGLGraph
from torch import nn

from SubgraphClassification.focal_loss import FocalLoss
from SubgraphClassification.GNN_encoder import GNN


def builder(
    g: DGLGraph,
    best_params: dict,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    device: torch.device
) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """
    Builds the complete model pipeline (model, optimizer, loss function) using best hyperparameters.

    Args:
        g (DGLGraph): DGL graph containing node features.
        best_params (dict): Dictionary containing the best hyperparameters from tuning.
        labels (Tensor): Label tensor for all nodes.
        train_idx (Tensor): Indices of training nodes.
        device (torch.device): Device to allocate the model and tensors.

    Returns:
        Tuple:
            - model (nn.Module): Instantiated GNN model.
            - optimizer (torch.optim.Optimizer): Optimizer configured with learning
             rate and weight decay.
            - loss_fn (nn.Module): Loss function, either CrossEntropyLoss or FocalLoss.
    """
    model = building_model(g, best_params, device)
    optimizer = building_optimizer(best_params, model)
    loss_fn = building_loss(best_params, labels, train_idx, device)
    return model, optimizer, loss_fn


def building_model(
    g: DGLGraph,
    best_params: dict,
    device: torch.device
) -> nn.Module:
    """
    Builds and returns a GNN model using the given hyperparameters.

    Args:
        g (DGLGraph): DGL graph used to infer input feature dimension.
        best_params (dict): Dictionary containing GNN architecture hyperparameters.
        device (torch.device): Device to place the model on.

    Returns:
        nn.Module: Initialized GNN model.
    """
    model = GNN(
        in_feats=g.ndata['feat'].shape[1],
        hidden_feats=best_params["hidden_feats"],
        num_layers=best_params["num_layers"],
        layer_type=best_params["layer_type"],
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
    best_params: dict,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    device: torch.device
) -> nn.Module:
    """
    Constructs and returns the appropriate loss function, either FocalLoss or
    class-weighted CrossEntropyLoss.

    Args:
        best_params (dict): Hyperparameters dict containing 'use_focal_loss'
         and optionally 'proportion'.
        labels (Tensor): Full label tensor.
        train_idx (Tensor): Tensor of training indices.
        device (torch.device): Device to place the loss weights on.

    Returns:
        nn.Module: Instantiated loss function (FocalLoss or CrossEntropyLoss).
    """
    if best_params["use_focal_loss"] is True:
        loss_fn = FocalLoss()
    else:
        # Compute class counts
        num_0 = (labels[train_idx] == 0).sum().item()
        num_1 = (labels[train_idx] == 1).sum().item()
        # Create class weights: higher for underrepresented class 1
        weights = torch.tensor(
            [1.0, (num_0 / num_1)*best_params["proportion"]], dtype=torch.float32
        )
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    return loss_fn
