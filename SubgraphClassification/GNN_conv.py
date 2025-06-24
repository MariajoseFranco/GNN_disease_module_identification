import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GNN, self).__init__()
        self.layer1 = dglnn.GraphConv(
            in_feats, hidden_feats, norm='both', allow_zero_in_degree=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_feats)

        self.layer2 = dglnn.GraphConv(
            hidden_feats, hidden_feats, norm='both', allow_zero_in_degree=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_feats)

        self.layer3 = dglnn.GraphConv(
            hidden_feats, hidden_feats, norm='both', allow_zero_in_degree=True
        )
        self.bn3 = nn.BatchNorm1d(hidden_feats)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_feats, 2)  # Binary classification (logits for 2 classes)

    def forward(self, g, features):
        h = self.layer1(g, features)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.layer2(g, h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.layer3(g, h)
        h = self.bn3(h)
        h = F.relu(h)
        h = self.dropout(h)

        return self.classifier(h)
