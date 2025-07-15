from typing import Tuple

import dgl
import numpy as np
import torch

from LinkPrediction.utils.helper_functions import edge_list_to_tensor


def splitting_sizes(
        G_dispro: dgl.DGLHeteroGraph, edge_type: Tuple[str, str, str]
) -> Tuple[np.ndarray, int, int, int]:
    """
    Generate shuffled edge indices and compute train/validation/test split
    sizes for a given edge type.

    Args:
        G_dispro (dgl.DGLHeteroGraph): The heterogeneous graph containing the edges to split.
        edge_type (Tuple[str, str, str]): The specific edge type to split
        (src_type, relation, dst_type).

    Returns:
        Tuple[np.ndarray, int, int, int]:
            - eids (np.ndarray): Shuffled edge indices.
            - train_size (int): Number of training edges (70%).
            - val_size (int): Number of validation edges (15%).
            - test_size (int): Number of test edges (15%).
    """
    eids = np.arange(G_dispro.num_edges(etype=edge_type))
    eids = np.random.permutation(eids)

    train_size = int(0.7 * len(eids))
    test_size = int(0.15 * len(eids))
    val_size = int(0.15 * len(eids))

    return eids, train_size, val_size, test_size


def pos_train_test_split(
    u: torch.Tensor,
    v: torch.Tensor,
    eids: np.ndarray,
    train_size: int,
    val_size: int,
    test_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Split positive edges into training, validation, and test sets for link prediction.

    Args:
        u (torch.Tensor): Source node indices of positive edges.
        v (torch.Tensor): Target node indices of positive edges.
        eids (np.ndarray): Shuffled array of edge indices.
        train_size (int): Number of positive edges for the training set.
        val_size (int): Number of positive edges for the validation set.
        test_size (int): Number of positive edges for the test set.

    Returns:
        Tuple:
            - train_pos_u (torch.Tensor): Training set source nodes.
            - train_pos_v (torch.Tensor): Training set target nodes.
            - val_pos_u (torch.Tensor): Validation set source nodes.
            - val_pos_v (torch.Tensor): Validation set target nodes.
            - test_pos_u (torch.Tensor): Test set source nodes.
            - test_pos_v (torch.Tensor): Test set target nodes.
            - val_eids (np.ndarray): Edge indices for validation set.
            - test_eids (np.ndarray): Edge indices for test set.
    """
    train_eids = eids[:train_size]
    val_eids = eids[train_size:train_size + val_size]
    test_eids = eids[train_size + val_size:]

    train_pos_u, train_pos_v = u[train_eids], v[train_eids]
    val_pos_u, val_pos_v = u[val_eids], v[val_eids]
    test_pos_u, test_pos_v = u[test_eids], v[test_eids]

    return train_pos_u, train_pos_v, val_pos_u, val_pos_v, test_pos_u, test_pos_v, val_eids, test_eids


def neg_train_test_split(
    g: dgl.DGLHeteroGraph,
    etype: tuple,
    train_size: int,
    val_size: int,
    test_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate negative samples for link prediction and split them into training,
    validation, and test sets.

    Negative samples are randomly generated node pairs that do not exist in the
    graph as positive edges.

    Args:
        g (dgl.DGLHeteroGraph): The heterogeneous graph.
        etype (tuple): Edge type (src_type, relation, dst_type).
        train_size (int): Number of negative samples for training.
        val_size (int): Number of negative samples for validation.
        test_size (int): Number of negative samples for testing.

    Returns:
        Tuple:
            - train_neg_u (torch.Tensor): Training set negative source nodes.
            - train_neg_v (torch.Tensor): Training set negative target nodes.
            - val_neg_u (torch.Tensor): Validation set negative source nodes.
            - val_neg_v (torch.Tensor): Validation set negative target nodes.
            - test_neg_u (torch.Tensor): Test set negative source nodes.
            - test_neg_v (torch.Tensor): Test set negative target nodes.
    """
    src_type, _, dst_type = etype

    # Existing positive edges
    existing_edges = set(zip(
        g.edges(etype=etype)[0].tolist(),
        g.edges(etype=etype)[1].tolist()
    ))

    total_neg_samples = g.num_edges(etype=etype)
    neg_edges = set()
    attempts = 0
    max_attempts = 100 * total_neg_samples  # Avoid infinite loops

    while len(neg_edges) < total_neg_samples and attempts < max_attempts:
        src = torch.randint(0, g.num_nodes(src_type), (1,)).item()
        dst = torch.randint(0, g.num_nodes(dst_type), (1,)).item()
        if (src, dst) not in existing_edges and (src, dst) not in neg_edges:
            neg_edges.add((src, dst))
        attempts += 1

    if len(neg_edges) < total_neg_samples:
        print(
            f"Warning: Only {len(neg_edges)} negative samples generated "
            f"out of {total_neg_samples} requested."
        )

    neg_edges = list(neg_edges)
    neg_edges = np.random.permutation(neg_edges)

    # Split into train, val, test
    train_edges = neg_edges[:train_size]
    val_edges = neg_edges[train_size:train_size + val_size]
    test_edges = neg_edges[train_size + val_size:]

    train_neg_u, train_neg_v = edge_list_to_tensor(train_edges)
    val_neg_u, val_neg_v = edge_list_to_tensor(val_edges)
    test_neg_u, test_neg_v = edge_list_to_tensor(test_edges)

    return train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_u, test_neg_v