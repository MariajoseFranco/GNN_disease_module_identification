import torch.nn as nn


class DotPredictor(nn.Module):
    def forward(self, g, h, u, v):
        return (h[u] * h[v]).sum(dim=1)
