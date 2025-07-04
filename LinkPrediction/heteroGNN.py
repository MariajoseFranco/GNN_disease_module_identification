import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class HeteroGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, etypes, node_types, num_layers=2, layer_type="SAGEConv", aggregator_type="mean", dropout=0.3):
        super(HeteroGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layer_type = layer_type
        self.aggregator_type = aggregator_type

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.node_types = node_types

        input_dim = in_feats
        for _ in range(num_layers):
            layer_dict = self.build_layer_dict(input_dim, hidden_feats, etypes)
            self.layers.append(dglnn.HeteroGraphConv(layer_dict, aggregate='sum'))

            # BatchNorm dinámico según node_types
            self.bns.append(nn.ModuleDict({
                ntype: nn.BatchNorm1d(hidden_feats) for ntype in self.node_types
            }))

            input_dim = hidden_feats

        # Proyección final por tipo de nodo
        self.final_proj = nn.ModuleDict({
            ntype: nn.Linear(hidden_feats, in_feats) for ntype in self.node_types
        })

    def build_layer_dict(self, in_feats, out_feats, etypes):
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

    def forward(self, g, inputs):  # inputs: dict of node features
        h = inputs
        for i in range(self.num_layers):
            h = self.layers[i](g, h)
            h = {
                ntype: self.dropout(F.relu(self.bns[i][ntype](h_ntype)))
                for ntype, h_ntype in h.items()
            }
        h = {
            ntype: self.final_proj[ntype](h_ntype)
            for ntype, h_ntype in h.items()
        }
        return h
