import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def encode(self, x, edge_index):
        print("Encoding...")
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_pairs):
        print("Decoding...")
        z1 = z[edge_pairs[0]]
        z2 = z[edge_pairs[1]]
        edge_feat = torch.cat([z1, z2], dim=1)
        return self.classifier(edge_feat).view(-1)

    def forward(self, x, edge_index, edge_pairs):
        print("Forward propagation...")
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs)
