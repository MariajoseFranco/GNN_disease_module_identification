import dgl
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


def convert_to_dgl_graph(ppi_graph: nx.Graph, seed_nodes: set, feature_dim=64):
    print("Converting NetworkX to DGLGraph")

    # Convert to DGL graph
    dgl_graph = dgl.from_networkx(ppi_graph)

    # Assign node features
    num_nodes = dgl_graph.num_nodes()
    node_features = torch.randn((num_nodes, feature_dim))  # or torch.eye(num_nodes)
    dgl_graph.ndata['feat'] = node_features

    # Assign binary labels: 1 if in seed_nodes, 0 otherwise
    id_map = {n: i for i, n in enumerate(ppi_graph.nodes())}
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for node in seed_nodes:
        if node in id_map:
            labels[id_map[node]] = 1
    dgl_graph.ndata['label'] = labels

    return dgl_graph


def edge_list_to_tensor(edge_list):
    return torch.tensor(edge_list, dtype=torch.long).T  # shape [2, E]
