import random

import torch


def generate_labels(
        seed_nodes: set[str], node_index_mapping: dict[str, int], num_nodes: int
) -> torch.Tensor:
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for node in seed_nodes:
        if node in node_index_mapping:
            labels[node_index_mapping[node]] = 1
    return labels


def generate_balanced_labels(
        seed_nodes: set[str], node_index_mapping: dict[str, int], all_nodes: list
):
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


def generate_expanded_labels(seed_nodes, node_index, all_nodes):
    labels = torch.zeros(len(all_nodes), dtype=torch.long)  # default to negative class

    for n in seed_nodes:
        if n in node_index:
            labels[node_index[n]] = 1  # mark positive

    return labels
