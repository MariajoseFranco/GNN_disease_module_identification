from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch


def edge_list_to_tensor(edge_list: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of edges into two torch tensors: source and destination nodes.

    Args:
        edge_list (List[Tuple[int, int]]): List of edges, where each edge is a
        (source, destination) tuple.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - u (torch.Tensor): Tensor of source node indices.
            - v (torch.Tensor): Tensor of destination node indices.
    """
    u = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
    v = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
    return u, v


def identifying_reverse_edges_to_remove(
        pos_u: torch.Tensor, pos_v: torch.Tensor
) -> Set[Tuple[int, int]]:
    """
    Identifies the set of reverse edges corresponding to a given set of positive edges.

    This function is useful for link prediction tasks to remove reverse edges
    during train/validation/test splits, avoiding information leakage.

    Args:
        pos_u (torch.Tensor): Tensor of source node indices of positive edges.
        pos_v (torch.Tensor): Tensor of target node indices of positive edges.

    Returns:
        Set[Tuple[int, int]]: A set of (dst, src) pairs representing reverse edges to remove.
    """
    return set(zip(pos_v.tolist(), pos_u.tolist()))


def obtaining_removing_reverse_edges_per_set(
    pos_u: torch.Tensor,
    pos_v: torch.Tensor,
    rev_u: torch.Tensor,
    rev_v: torch.Tensor
) -> List[int]:
    """
    Identifies the edge indices in the reverse graph that correspond to a set of positive edges.

    Args:
        pos_u (torch.Tensor): Source nodes of positive edges.
        pos_v (torch.Tensor): Destination nodes of positive edges.
        rev_u (torch.Tensor): Source nodes of reverse edges.
        rev_v (torch.Tensor): Destination nodes of reverse edges.

    Returns:
        List[int]: List of indices of reverse edges to remove from the graph.
    """
    pairs_set = identifying_reverse_edges_to_remove(pos_u, pos_v)
    rev_eids_to_remove = [
        i for i, (src, dst) in enumerate(zip(rev_u.tolist(), rev_v.tolist()))
        if (src, dst) in pairs_set
    ]
    return rev_eids_to_remove


def obtaining_removing_reverse_edges(
    val_pos_u: torch.Tensor,
    val_pos_v: torch.Tensor,
    test_pos_u: torch.Tensor,
    test_pos_v: torch.Tensor,
    rev_u: torch.Tensor,
    rev_v: torch.Tensor
) -> np.ndarray:
    """
    Identifies and concatenates the indices of reverse edges to remove for
    both validation and test sets.

    Args:
        val_pos_u (torch.Tensor): Source nodes of validation positive edges.
        val_pos_v (torch.Tensor): Destination nodes of validation positive edges.
        test_pos_u (torch.Tensor): Source nodes of test positive edges.
        test_pos_v (torch.Tensor): Destination nodes of test positive edges.
        rev_u (torch.Tensor): Source nodes of reverse edges.
        rev_v (torch.Tensor): Destination nodes of reverse edges.

    Returns:
        np.ndarray: Array of reverse edge indices to remove.
    """
    rev_val_eids_to_remove = obtaining_removing_reverse_edges_per_set(
        val_pos_u, val_pos_v, rev_u, rev_v
    )
    rev_test_eids_to_remove = obtaining_removing_reverse_edges_per_set(
        test_pos_u, test_pos_v, rev_u, rev_v
    )
    rev_eids_to_remove = np.concatenate(
        [rev_val_eids_to_remove, rev_test_eids_to_remove]
    )
    return rev_eids_to_remove


def getting_seed_nodes(
    df_dis_pro: pd.DataFrame,
    selected_diseases: List[str]
) -> Dict[str, Tuple[str]]:
    """
    Extracts the seed proteins associated with each selected disease.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing disease–protein associations.
            Must include columns 'disease_name' and 'protein_id'.
        selected_diseases (List[str]): List of disease names to extract seed proteins for.

    Returns:
        Dict[str, Tuple[str]]: Dictionary mapping each disease to a tuple of its
        associated protein IDs (seed nodes).
    """
    seed_nodes = {}
    for disease in selected_diseases:
        df = df_dis_pro[df_dis_pro['disease_name'] == disease]
        tuple_seed_nodes = tuple(df['protein_id'])
        seed_nodes[disease] = tuple_seed_nodes
    return seed_nodes
