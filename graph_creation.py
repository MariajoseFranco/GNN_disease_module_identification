import dgl
import networkx as nx
import pandas as pd
import torch
from networkx import Graph


class GraphPPI():
    def __init__(self):
        pass

    def create_homogeneous_graph(self, df_pro_pro: pd.DataFrame) -> Graph:
        """
        Create a graph from the PPI data

        Args:
            df_pro_pro: dataframe that contains the protein-protein interaction

        Returns:
            Graph: graph created from this PPI
        """
        print('Creating Protein-Protein Graph')
        G_ppi = nx.from_pandas_edgelist(df_pro_pro, 'prA', 'prB')
        print(f"PPI Network: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")
        print('...done')
        return G_ppi

    def create_heterogeneous_graph(self, df_dis_pro, df_pro_pro):
        print('Creating Disease-Protein Graph')
        # Build heterograph dictionary
        data_dict = {
            ('protein', 'interacts', 'protein'): (
                df_pro_pro['src_id'].values, df_pro_pro['dst_id'].values
            ),
            ('disease', 'associates', 'protein'): (
                df_dis_pro['disease_id'].values, df_dis_pro['protein_id_enc'].values
            ),
            ('protein', 'rev_associates', 'disease'): (
                df_dis_pro['protein_id_enc'].values, df_dis_pro['disease_id'].values
            )
        }

        # Create heterograph
        hetero_graph = dgl.heterograph(data_dict)
        print('...done')
        return hetero_graph

    def convert_networkx_to_dgl_graph(self, ppi_graph: nx.Graph, seed_nodes: set, feature_dim=64):
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
        print("...done")
        return dgl_graph

    def convert_to_tensors(self, train_u, train_v, test_u, test_v):
        train_u = torch.tensor(train_u, dtype=torch.long)
        train_v = torch.tensor(train_v, dtype=torch.long)
        test_u = torch.tensor(test_u, dtype=torch.long)
        test_v = torch.tensor(test_v, dtype=torch.long)
        return train_u, train_v, test_u, test_v

    def convert_to_heterogeneous_graph(self, G_dispro, edge_type, u, v):
        g = dgl.heterograph(
            {edge_type: (u, v)},
            num_nodes_dict={
                'disease': G_dispro.num_nodes('disease'),
                'protein': G_dispro.num_nodes('protein')
            }
        )
        return g
