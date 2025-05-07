import random

import networkx as nx
import torch
from torch_geometric.data import Data


def generate_data(ppi_graph: nx.Graph, feature_dim=64):
    print("Prepare torch_geometric Data object from PPI graph")
    node_list = list(ppi_graph.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}

    edge_index = torch.tensor(
        [[node_index[u], node_index[v]] for u, v in ppi_graph.edges()], dtype=torch.long
    ).t().contiguous()

    x = torch.eye(
        len(node_list)
    ) if feature_dim == len(node_list) else torch.randn((len(node_list), feature_dim))
    print("...done")
    return Data(x=x, edge_index=edge_index), node_index


def create_edge_labels(edge_index, num_nodes, num_neg_samples=1):
    print("Create edge and labels")
    pos_edges = edge_index.t().tolist()
    neg_edges = set()
    while len(neg_edges) < len(pos_edges) * num_neg_samples:
        u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if u != v and [u, v] not in pos_edges and [v, u] not in pos_edges:
            neg_edges.add((u, v))

    all_edges = pos_edges + list(neg_edges)
    edge_tensor = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    labels = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))])
    print("...done")
    return edge_tensor, labels
