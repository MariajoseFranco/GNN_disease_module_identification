import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers=3, layer_type="SAGEConv", dropout=0.3):
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

        # Final classifier
        self.classifier = nn.Linear(hidden_feats, 2)

    def forward(self, g, features):
        h = features
        for layer, bn in zip(self.layers, self.bns):
            h = layer(g, h)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

        return self.classifier(h)

    def get_layer_class(self, layer_type):
        if layer_type == "SAGEConv":
            def layer(in_f, out_f):
                return dglnn.SAGEConv(in_f, out_f, aggregator_type='mean')
        elif layer_type == "GraphConv":
            def layer(in_f, out_f):
                return dglnn.GraphConv(in_f, out_f)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        return layer
