import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class HeteroGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers=2, layer_type="SAGEConv", aggregator_type="mean", dropout=0.3):
        super(HeteroGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Capa de entrada
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layer_type = layer_type
        self.aggregator_type = aggregator_type

        input_dim = in_feats
        for i in range(num_layers):
            layer_dict = self.build_layer_dict(input_dim, hidden_feats)
            self.layers.append(dglnn.HeteroGraphConv(layer_dict, aggregate='sum'))
            # BatchNorm por tipo de nodo (asumiendo 'disease' y 'protein')
            self.bns.append(nn.ModuleDict({
                'disease': nn.BatchNorm1d(hidden_feats),
                'protein': nn.BatchNorm1d(hidden_feats)
            }))
            input_dim = hidden_feats  # Para capas siguientes

        self.final_proj = nn.ModuleDict({
            'disease': nn.Linear(hidden_feats, in_feats),
            'protein': nn.Linear(hidden_feats, in_feats)
        })

    def build_layer_dict(self, in_feats, out_feats):
        if self.layer_type == "SAGEConv":
            return {
                'associates': dglnn.SAGEConv(in_feats, out_feats, aggregator_type=self.aggregator_type),
                'rev_associates': dglnn.SAGEConv(in_feats, out_feats, aggregator_type=self.aggregator_type),
                'interacts': dglnn.SAGEConv(in_feats, out_feats, aggregator_type=self.aggregator_type),
                'rev_interacts': dglnn.SAGEConv(in_feats, out_feats, aggregator_type=self.aggregator_type),
            }
        elif self.layer_type == "GraphConv":
            return {
                'associates': dglnn.GraphConv(in_feats, out_feats),
                'rev_associates': dglnn.GraphConv(in_feats, out_feats),
                'interacts': dglnn.GraphConv(in_feats, out_feats),
                'rev_interacts': dglnn.GraphConv(in_feats, out_feats),
            }
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

    def forward(self, g, inputs):  # inputs: dict of node features
        h = inputs
        for i in range(self.num_layers):
            h = self.layers[i](g, h)
            # Aplicar BatchNorm, ReLU y Dropout por tipo de nodo
            h = {
                ntype: self.dropout(F.relu(self.bns[i][ntype](h_ntype)))
                for ntype, h_ntype in h.items()
            }
        h = {
            ntype: self.final_proj[ntype](h_ntype)
            for ntype, h_ntype in h.items()
        }
        return h
