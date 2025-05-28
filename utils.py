import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from matplotlib.patches import Patch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        parts = re.split(r'[^a-zA-Z0-9]+', token)
        for part in parts:
            if part:  # Skip empty strings
                cleaned_tokens.append(part.lower())
    filtered_tokens = [token for token in cleaned_tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens


# Mapping Functions

def mapping_diseases_to_proteins(df_dis_pro: pd.DataFrame) -> dict:
    """
    Map the disease to the proteins that are associated with it

    Args:
        df_dis_pro: dataframe containing information about
        the interaction between proteins and diseases

    Returns:
        dict: dictionary where the keys are the diseases of
        interest and the value for each key (disease) is the
        set of proteins that are present in that disease
    """
    disease_pro_mapping = df_dis_pro.groupby("disease_name")["protein_id"].apply(set).to_dict()
    return disease_pro_mapping


def mapping_index_to_node(df_dis_pro, df_pro_pro):
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


# Train/Test Split Functions

def pos_train_test_split(u, v, eids, test_size):
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    return train_pos_u, train_pos_v, test_pos_u, test_pos_v


def neg_train_test_split_gnn(g, etype, num_samples, test_size):
    src_type, _, dst_type = g.to_canonical_etype(etype)

    # Sample negative candidate edges
    src_ids = torch.randint(0, g.num_nodes(src_type), (num_samples,))
    dst_ids = torch.randint(0, g.num_nodes(dst_type), (num_samples,))

    # Ensure test_size < num_samples
    assert test_size < num_samples, "test_size must be smaller than num_samples"

    # Shuffle indices for splitting
    indices = np.random.permutation(num_samples)

    # Train/Test split
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    test_neg_u = src_ids[test_indices]
    test_neg_v = dst_ids[test_indices]
    train_neg_u = src_ids[train_indices]
    train_neg_v = dst_ids[train_indices]

    return train_neg_u, train_neg_v, test_neg_u, test_neg_v


def neg_train_test_split_subg(u, v, g, test_size):
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.num_edges())
    test_neg_u, test_neg_v = (
        neg_u[neg_eids[:test_size]],
        neg_v[neg_eids[:test_size]],
    )
    train_neg_u, train_neg_v = (
        neg_u[neg_eids[test_size:]],
        neg_v[neg_eids[test_size:]],
    )
    return train_neg_u, train_neg_v, test_neg_u, test_neg_v
