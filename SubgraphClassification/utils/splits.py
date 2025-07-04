from typing import Tuple

import torch


def split_train_test_val_indices(
    labels: torch.Tensor, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits labeled nodes into balanced train, validation, and test indices.

    The dataset is shuffled and split into train, validation, and test sets using
    the provided ratios. Within the training set, oversampling is applied to positive
    samples (label = 1) to reduce class imbalance.

    Args:
        labels (torch.Tensor): Tensor of shape (N,) with binary labels (0 or 1).
        train_ratio (float): Proportion of data to use for training. Default is 0.7.
        val_ratio (float): Proportion of data to use for validation. Default is 0.15.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - train_idx_balanced: Indices of the balanced training set
            - val_idx: Indices for the validation set
            - test_idx: Indices for the test set
    """
    labeled_idx = torch.arange(len(labels))
    shuffled = labeled_idx[torch.randperm(len(labeled_idx))]

    n = len(shuffled)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:n_train + n_val]
    test_idx = shuffled[n_train + n_val:]

    pos_train_idx = train_idx[labels[train_idx] == 1]
    neg_train_idx = train_idx[labels[train_idx] == 0]

    desired_pos_size = len(neg_train_idx) // 2  # Adjust ratio if needed
    if len(pos_train_idx) > 0:
        pos_oversampled = pos_train_idx[torch.randint(0, len(pos_train_idx), (desired_pos_size,))]
    else:
        pos_oversampled = pos_train_idx

    train_idx_balanced = torch.cat([neg_train_idx, pos_oversampled])
    train_idx_balanced = train_idx_balanced[torch.randperm(len(train_idx_balanced))]

    return train_idx_balanced, val_idx, test_idx
