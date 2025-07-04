import random

import torch


def generate_labels(
    seed_nodes: set[str],
    node_index_mapping: dict[str, int],
    num_nodes: int
) -> torch.Tensor:
    """
    Generates a binary label tensor for node classification.

    Args:
        seed_nodes (set[str]): Set of protein nodes associated with the disease.
        node_index_mapping (dict[str, int]): Mapping from node names to index positions.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        torch.Tensor: Tensor of shape (num_nodes,) where each position is:
            - 1 if the node is a seed node (positive)
            - 0 otherwise (negative)
    """
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for node in seed_nodes:
        if node in node_index_mapping:
            labels[node_index_mapping[node]] = 1
    return labels


def generate_balanced_labels(
    seed_nodes: set[str],
    node_index_mapping: dict[str, int],
    all_nodes: list[str]
) -> torch.Tensor:
    """
    Generates labels with a balanced number of positive and negative samples.
    Unlabeled nodes are initially ignored (-1), and a random sample of negatives
    is selected to match the number of positives.

    Args:
        seed_nodes (set[str]): Set of known positive nodes.
        node_index_mapping (dict[str, int]): Mapping from node name to index.
        all_nodes (list[str]): List of all node names in the subgraph.

    Returns:
        torch.Tensor: Tensor of shape (len(all_nodes),) with labels:
            - 1 for seed nodes (positives)
            - 0 for sampled negatives
            - -1 for ignored nodes
    """
    labels = torch.full((len(all_nodes),), -1, dtype=torch.long)  # -1: ignore

    # Set positive labels
    positive_indices = [
        node_index_mapping[node] for node in seed_nodes if node in node_index_mapping
    ]
    labels[positive_indices] = 1

    # Sample balanced negatives
    unlabeled = [i for i in range(len(all_nodes)) if i not in positive_indices]
    sampled_negatives = torch.tensor(
        random.sample(unlabeled, len(positive_indices)), dtype=torch.long
    )
    labels[sampled_negatives] = 0
    return labels


def generate_expanded_labels(
    seed_nodes: set[str],
    node_index: dict[str, int],
    all_nodes: list[str]
) -> torch.Tensor:
    """
    Generates binary classification labels for nodes based on known seed nodes.

    Args:
        seed_nodes (set[str]): Known disease-associated proteins (positives).
        node_index (dict[str, int]): Mapping from node names to index positions.
        all_nodes (list[str]): List of all nodes in the subgraph.

    Returns:
        torch.Tensor: Tensor of shape (len(all_nodes),) where:
            - 1 indicates a positive (seed node)
            - 0 indicates a negative (non-seed node)
    """
    labels = torch.zeros(len(all_nodes), dtype=torch.long)  # default to negative class

    for n in seed_nodes:
        if n in node_index:
            labels[node_index[n]] = 1  # mark positive

    return labels
