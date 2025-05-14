import dgl.nn as dglnn
import torch
import torch.nn.functional as F
from torch import nn


class GNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)
        )

    def encode(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

    def decode(self, z, u, v):
        h_uv = torch.cat([z[u], z[v]], dim=1)
        return self.decoder(h_uv).squeeze()

    def forward(self, g, features, u, v):
        z = self.encode(g, features)
        return self.decode(z, u, v)
