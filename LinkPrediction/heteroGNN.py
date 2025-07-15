from typing import Dict, List, Tuple

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for Node Embedding Learning.

    This model supports multiple GNN layer types (GraphConv or SAGEConv) over a heterogeneous graph,
    with per-node-type BatchNorm, activation, dropout, and final projection layers.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        etypes: List[Tuple[str, str, str]],
        node_types: List[str],
        num_layers: int = 2,
        layer_type: str = "SAGEConv",
        aggregator_type: str = "mean",
        dropout: float = 0.3
    ):
        """
        Initialize the HeteroGNN model.

        Args:
            in_feats (int): Input feature dimension for all node types.
            hidden_feats (int): Hidden feature dimension after GNN layers.
            etypes (List[Tuple[str, str, str]]): List of edge types
            (canonical edge format) used in the graph.
            node_types (List[str]): List of node types present in the graph.
            num_layers (int, optional): Number of GNN layers. Default is 2.
            layer_type (str, optional): GNN layer type ('SAGEConv' or 'GraphConv').
            Default is 'SAGEConv'.
            aggregator_type (str, optional): Aggregator type for SAGEConv
            ('mean', 'pool', 'lstm', etc.). Default is 'mean'.
            dropout (float, optional): Dropout rate applied after each layer. Default is 0.3.
        """
        super(HeteroGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layer_type = layer_type
        self.aggregator_type = aggregator_type
        self.node_types = node_types

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        input_dim = in_feats
        for _ in range(num_layers):
            layer_dict = self.build_layer_dict(input_dim, hidden_feats, etypes)
            self.layers.append(dglnn.HeteroGraphConv(layer_dict, aggregate='sum'))

            # BatchNorm per node type
            self.bns.append(nn.ModuleDict({
                ntype: nn.BatchNorm1d(hidden_feats) for ntype in self.node_types
            }))

            input_dim = hidden_feats

        # Final projection to original in_feats dimension per node type
        self.final_proj = nn.ModuleDict({
            ntype: nn.Linear(hidden_feats, in_feats) for ntype in self.node_types
        })

    def build_layer_dict(
        self,
        in_feats: int,
        out_feats: int,
        etypes: List[Tuple[str, str, str]]
    ) -> Dict[Tuple[str, str, str], nn.Module]:
        """
        Builds a dictionary of GNN layers per edge type.

        Args:
            in_feats (int): Input feature size.
            out_feats (int): Output feature size.
            etypes (List[Tuple[str, str, str]]): List of edge types.

        Returns:
            Dict[Tuple[str, str, str], nn.Module]: Dictionary mapping edge types to GNN layers.
        """
        if self.layer_type == "SAGEConv":
            return {
                etype: dglnn.SAGEConv(in_feats, out_feats, aggregator_type=self.aggregator_type)
                for etype in etypes
            }
        elif self.layer_type == "GraphConv":
            return {
                etype: dglnn.GraphConv(in_feats, out_feats)
                for etype in etypes
            }
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

    def forward(
            self, g: dgl.DGLHeteroGraph, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the HeteroGNN.

        Args:
            g (dgl.DGLHeteroGraph): The heterogeneous input graph.
            inputs (dict): Input node features by node type. Dict[str, Tensor].

        Returns:
            dict: Output node embeddings by node type after GNN and final projection layers.
        """
        h = inputs
        for i in range(self.num_layers):
            h = self.layers[i](g, h)
            h = {
                ntype: self.dropout(F.relu(self.bns[i][ntype](h_ntype)))
                for ntype, h_ntype in h.items()
            }
        # Apply final projection layer per node type
        h = {
            ntype: self.final_proj[ntype](h_ntype)
            for ntype, h_ntype in h.items()
        }
        return h
