import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Patch


def load_config(config_path='config.yaml'):
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file. Defaults to 'config.yaml'.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Mapping Functions

def mapping_diseases_to_proteins(df_dis_pro: pd.DataFrame) -> dict:
    """
    Map each disease to its associated proteins.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame containing disease-protein associations.

    Returns:
        dict: Dictionary mapping disease names to sets of associated proteins.
    """
    disease_pro_mapping = {
        disease: dict(zip(group['protein_id'], group['score']))
        for disease, group in df_dis_pro.groupby("disease_name")
    }
    return disease_pro_mapping


def mapping_index_to_node(df_dis_pro, df_pro_pro):
    """
    Create dictionaries mapping node indices to node names for diseases and proteins.

    Args:
        df_dis_pro (pd.DataFrame): DataFrame with disease-protein associations and encoded IDs.
        df_pro_pro (pd.DataFrame): DataFrame with protein-protein interactions and encoded IDs.

    Returns:
        tuple:
            - disease_index_to_node (dict): Maps disease IDs to disease names.
            - protein_index_to_node (dict): Maps protein IDs to protein names.
    """
    # Disease mapping
    disease_index_to_node = {
        idx: disease for idx, disease in df_dis_pro[['disease_id', 'disease_name']]
        .drop_duplicates()
        .values
    }

    # Protein mapping
    # Extract and unify mappings from all sources
    dis_pro_mapping = df_dis_pro[['protein_id_enc', 'protein_id']].drop_duplicates()
    src_mapping = df_pro_pro[['src_id', 'prA']].drop_duplicates()
    dst_mapping = df_pro_pro[['dst_id', 'prB']].drop_duplicates()

    # Rename columns for consistency
    src_mapping.columns = ['protein_id_enc', 'protein_id']
    dst_mapping.columns = ['protein_id_enc', 'protein_id']

    # Combine all mappings
    combined_mapping = pd.concat([dis_pro_mapping, src_mapping, dst_mapping]).drop_duplicates()

    # Create dictionary
    protein_index_to_node = {
        idx: protein for idx, protein in combined_mapping.values
    }
    return disease_index_to_node, protein_index_to_node


# Visualization Functions

def visualize_disease_protein_associations(g, diseases, max_edges=200):
    """
    Visualize disease-protein associations from a heterogeneous graph using NetworkX and Matplotlib.

    Args:
        g (dgl.DGLHeteroGraph): The heterogeneous graph containing disease-protein associations.
        diseases (list): List of disease node indices to visualize.
        max_edges (int): Maximum number of edges to plot. Defaults to 200.

    Returns:
        None
    """
    # Only use 'associates' edge type
    etype = ('disease', 'associates', 'protein')
    src, dst = g.edges(etype=etype)

    # Filter edges for the selected disease nodes
    mask = torch.isin(src, torch.tensor(diseases))
    src = src[mask]
    dst = dst[mask]

    # Optionally limit number of edges
    if len(src) > max_edges:
        indices = torch.randperm(len(src))[:max_edges]
        src = src[indices]
        dst = dst[indices]

    # Build a NetworkX graph
    G_nx = nx.DiGraph()
    for s, d in zip(src.tolist(), dst.tolist()):
        disease_label = f"disease_{s}"
        protein_label = f"protein_{d}"
        G_nx.add_node(disease_label, bipartite=0)
        G_nx.add_node(protein_label, bipartite=1)
        G_nx.add_edge(disease_label, protein_label)

    # Layout and draw
    pos = nx.spring_layout(G_nx, k=0.5, seed=42)
    plt.figure(figsize=(16, 10))
    node_colors = ['lightcoral' if n.startswith('disease') else 'skyblue' for n in G_nx.nodes()]
    nx.draw(
        G_nx, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=300,
        font_size=7,
        edge_color='gray',
        alpha=0.9
    )

    # Legend
    legend_elements = [
        Patch(facecolor='lightcoral', label='Diseases'),
        Patch(facecolor='skyblue', label='Proteins')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Disease–Protein Associations")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Obtaining Labels Function

def generate_labels(
        seed_nodes: set[str], node_index_mapping: dict[str, int], num_nodes: int
) -> torch.Tensor:
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for node in seed_nodes:
        if node in node_index_mapping:
            labels[node_index_mapping[node]] = 1
    return labels


# Train/Test Split Functions

def split_train_test_val_indices(
        node_index_mapping: dict[str, int], train_ratio=0.7, val_ratio=0.15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_nodes = list(node_index_mapping.values())
    random.shuffle(all_nodes)
    n = len(all_nodes)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    return (
        torch.tensor(all_nodes[:train_end]),
        torch.tensor(all_nodes[train_end:val_end]),
        torch.tensor(all_nodes[val_end:])
    )


def pos_train_test_split(u, v, eids, train_size, val_size, test_size):
    """
    Split positive edges into training and test sets for link prediction.

    Args:
        u (Tensor): Source node indices of positive edges.
        v (Tensor): Target node indices of positive edges.
        eids (array-like): Array of shuffled edge indices.
        test_size (int): Number of edges to include in the test set.

    Returns:
        tuple: (train_pos_u, train_pos_v, test_pos_u, test_pos_v)
    """
    train_eids = eids[:train_size]
    val_eids = eids[train_size:train_size + val_size]
    test_eids = eids[train_size + val_size:]

    train_pos_u, train_pos_v = u[train_eids], v[train_eids]
    val_pos_u, val_pos_v = u[val_eids], v[val_eids]
    test_pos_u, test_pos_v = u[test_eids], v[test_eids]
    return train_pos_u, train_pos_v, val_pos_u, val_pos_v, test_pos_u, test_pos_v, val_eids, test_eids


def neg_train_test_split(g, etype, train_size, val_size, test_size):
    """
    Generate and split negative samples for a heterogeneous graph (DGL format).

    Args:
        g (dgl.DGLHeteroGraph): The heterogeneous graph.
        etype (tuple): The edge type tuple (src_type, relation, dst_type).
        num_samples (int): Number of negative samples to generate.
        test_size (int): Number of negative edges to include in the test set.

    Returns:
        tuple: (train_neg_u, train_neg_v, test_neg_u, test_neg_v)
    """
    src_type, _, dst_type = g.to_canonical_etype(etype)

    # Get existing edges as a set
    existing_edges = set(zip(
        g.edges(etype=etype)[0].tolist(),
        g.edges(etype=etype)[1].tolist()
    ))

    total_neg_samples = train_size + val_size + test_size
    neg_edges = set()
    attempts = 0
    max_attempts = 100 * total_neg_samples

    while len(neg_edges) < total_neg_samples and attempts < max_attempts:
        src = torch.randint(0, g.num_nodes(src_type), (1,)).item()
        dst = torch.randint(0, g.num_nodes(dst_type), (1,)).item()
        if (src, dst) not in existing_edges and (src, dst) not in neg_edges:
            neg_edges.add((src, dst))
        attempts += 1

    if len(neg_edges) < total_neg_samples:
        print(
            f"Warning: Only {len(neg_edges)} negative samples generated"
            f"out of {total_neg_samples} requested."
        )

    neg_edges = list(neg_edges)
    neg_edges = np.random.permutation(neg_edges)

    # Split
    train_edges = neg_edges[:train_size]
    val_edges = neg_edges[train_size:train_size + val_size]
    test_edges = neg_edges[train_size + val_size:]

    train_neg_u, train_neg_v = edge_list_to_tensor(train_edges)
    val_neg_u, val_neg_v = edge_list_to_tensor(val_edges)
    test_neg_u, test_neg_v = edge_list_to_tensor(test_edges)

    return train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_u, test_neg_v


def edge_list_to_tensor(edge_list):
    u = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
    v = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
    return u, v
