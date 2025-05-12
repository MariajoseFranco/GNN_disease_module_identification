import dgl.nn as dglnn
import torch
import torch.nn.functional as F
from torch import nn


class GNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, hidden_feats)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)
        )

    def encode(self, g, x):
        h = self.conv1(g, x)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

    def decode(self, z, u, v):
        # Concatenate node embeddings of edge endpoints
        z_uv = torch.cat([z[u], z[v]], dim=1)
        return self.classifier(z_uv).squeeze()

    def forward(self, g, x, u, v):
        z = self.encode(g, x)
        return self.decode(z, u, v)
