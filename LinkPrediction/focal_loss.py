import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.

    This loss is designed to address class imbalance by focusing training on hard examples.
    It reduces the loss contribution from easy samples and extends cross-entropy.

    Args:
        alpha (float, optional): Balancing factor for the positive class. Default is 0.25.
        gamma (float, optional): Focusing parameter to reduce easy examples' contribution.
        Default is 2.0.
        reduction (str, optional): Specifies reduction to apply to the output: 'mean' | 'sum'.
        Default is 'mean'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the focal loss between predicted logits and ground truth labels.

        Args:
            logits (Tensor): Predicted unnormalized logits, shape (N,).
            targets (Tensor): Ground truth binary labels (0 or 1), shape (N,).

        Returns:
            Tensor: Computed focal loss (scalar).
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Compute binary cross-entropy per example
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute the modulating factor
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma

        # Final focal loss computation
        loss = alpha_t * focal_factor * bce

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
