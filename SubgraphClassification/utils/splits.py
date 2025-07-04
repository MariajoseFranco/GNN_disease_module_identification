import torch


def split_train_test_val_indices(labels, train_ratio=0.7, val_ratio=0.15):
    labeled_idx = torch.arange(len(labels))  # all nodes are now labeled 0 or 1
    shuffled = labeled_idx[torch.randperm(len(labeled_idx))]

    n = len(shuffled)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:n_train + n_val]
    test_idx = shuffled[n_train + n_val:]

    # Identify positive and negative indices within train set
    pos_train_idx = train_idx[labels[train_idx] == 1]
    neg_train_idx = train_idx[labels[train_idx] == 0]

    # Oversample positives (with replacement)
    desired_pos_size = len(neg_train_idx) // 2  # or another ratio
    if len(pos_train_idx) > 0:
        pos_oversampled = pos_train_idx[torch.randint(0, len(pos_train_idx), (desired_pos_size,))]
    else:
        pos_oversampled = pos_train_idx  # fallback if no positives

    # Concatenate new training index
    train_idx_balanced = torch.cat([neg_train_idx, pos_oversampled])
    train_idx_balanced = train_idx_balanced[torch.randperm(len(train_idx_balanced))]

    return train_idx_balanced, val_idx, test_idx
