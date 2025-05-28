import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class HeteroGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(HeteroGNN, self).__init__()
        self.layers1 = dglnn.HeteroGraphConv({
            'associates': dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type='mean'),
            'rev_associates': dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type='mean'),
            'interacts': dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type='mean'),
        }, aggregate='sum')

        self.layers2 = dglnn.HeteroGraphConv({
            'associates': dglnn.SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean'),
            'rev_associates': dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type='mean'),
            'interacts': dglnn.SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean'),
        }, aggregate='sum')

    def forward(self, g, inputs):  # inputs is a dict of node features
        """
        Perform a forward pass of the Heterogeneous GNN model.

        The model applies two layers of heterogeneous graph convolutions (SAGEConv),
        using ReLU activations after the first layer.

        Args:
            g (dgl.DGLHeteroGraph): The input heterogeneous graph.
            inputs (dict): A dictionary mapping node types (e.g., 'disease', 'protein')
                        to their corresponding feature tensors.

        Returns:
            dict: A dictionary of output node embeddings, with keys being node types and
                values being the corresponding feature tensors after the second GNN layer.
        """
        h = self.layers1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.layers2(g, h)
        return h
