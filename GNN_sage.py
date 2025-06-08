import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, features, hidden_feats):
        super(GNN, self).__init__()
        self.layer1 = dglnn.SAGEConv(features, hidden_feats, aggregator_type='mean')
        self.layer2 = dglnn.SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        self.classifier = nn.Linear(hidden_feats, 2)  # <-- binary classification

    def forward(self, g, features):
        """
        Forward pass of the GNN model using GraphSAGE convolution layers.

        This method applies two layers of GraphSAGE convolution with a ReLU activation in between.
        It takes the input node features, performs message passing, and returns the updated
        node representations.

        Args:
            g (dgl.DGLGraph): The input graph.
            in_feat (torch.Tensor): Input node features of shape (num_nodes, in_features).

        Returns:
            torch.Tensor: Updated node features of shape (num_nodes, hidden_features).
        """
        h = self.layer1(g, features)
        h = F.relu(h)
        h = self.layer2(g, h)
        return self.classifier(h)
