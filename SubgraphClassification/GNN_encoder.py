from typing import Callable

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    """
    General Graph Neural Network (GNN) model supporting different layer types
    and flexible depth.

    The model consists of multiple graph convolution layers followed by batch
    normalization, ReLU activations, dropout regularization, and a final linear
    classifier for binary classification tasks.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        num_layers: int = 3,
        layer_type: str = "SAGEConv",
        dropout: float = 0.3
    ) -> None:
        """
        Initializes the GNN model.

        Args:
            in_feats (int): Dimension of input node features.
            hidden_feats (int): Dimension of hidden layer features.
            num_layers (int, optional): Number of GNN layers. Default is 3.
            layer_type (str, optional): Type of GNN layer to use ('SAGEConv' or 'GraphConv').
             Default is 'SAGEConv'.
            dropout (float, optional): Dropout rate applied after each layer. Default is 0.3.
        """
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        layer_class = self.get_layer_class(layer_type)

        # First layer
        self.layers.append(layer_class(in_feats, hidden_feats))
        self.bns.append(nn.BatchNorm1d(hidden_feats))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(layer_class(hidden_feats, hidden_feats))
            self.bns.append(nn.BatchNorm1d(hidden_feats))

        # Output classifier for binary classification (2 classes)
        self.classifier = nn.Linear(hidden_feats, 2)

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN model.

        Args:
            g (dgl.DGLGraph): The input graph.
            features (torch.Tensor): Node feature matrix of shape (N, in_feats).

        Returns:
            torch.Tensor: Logits for binary classification, shape (N, 2).
        """
        h = features
        for layer, bn in zip(self.layers, self.bns):
            h = layer(g, h)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

        return self.classifier(h)

    def get_layer_class(self, layer_type: str) -> Callable[[int, int], nn.Module]:
        """
        Returns a graph convolution layer class based on the specified layer type.

        Args:
            layer_type (str): Type of graph convolution ('SAGEConv' or 'GraphConv').

        Returns:
            Callable[[int, int], nn.Module]: A function that constructs a layer given
             input/output dimensions.

        Raises:
            ValueError: If the specified layer_type is not supported.
        """
        if layer_type == "SAGEConv":
            def layer(in_f: int, out_f: int) -> nn.Module:
                return dglnn.SAGEConv(in_f, out_f, aggregator_type='mean')
        elif layer_type == "GraphConv":
            def layer(in_f: int, out_f: int) -> nn.Module:
                return dglnn.GraphConv(in_f, out_f)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        return layer
