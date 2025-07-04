import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks with class imbalance.

    This loss function down-weights easy examples and focuses learning on hard,
    misclassified examples.
    Commonly used in scenarios with severe class imbalance.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """
        Initializes the FocalLoss module.

        Args:
            alpha (float, optional): Weighting factor for class imbalance. Default is 0.25.
            gamma (float, optional): Focusing parameter to reduce the relative loss for
             well-classified examples. Default is 2.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the focal loss between logits and ground truth targets.

        Args:
            logits (torch.Tensor): Predicted unnormalized scores (logits), shape (N, C)
             where C is the number of classes.
            targets (torch.Tensor): Ground truth class indices, shape (N,).

        Returns:
            torch.Tensor: Scalar tensor representing the mean focal loss over the batch.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
