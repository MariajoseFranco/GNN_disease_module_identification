import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal loss for binary classification.

        Args:
            alpha (float): Weighting factor for class 1 (positive class).
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): 'mean' or 'sum' over the batch.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Predicted logits of shape (N,).
            targets (Tensor): Ground truth binary labels (0 or 1), shape (N,).
        Returns:
            Tensor: Computed focal loss.
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Focal loss components
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma

        loss = alpha_t * focal_factor * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()
