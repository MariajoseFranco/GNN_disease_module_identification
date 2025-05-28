import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GNN, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')

    def forward(self, g, in_feat):
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
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
