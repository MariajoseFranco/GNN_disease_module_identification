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
        Create a homogeneous graph from protein-protein interaction data.

        Args:
            df_pro_pro (pd.DataFrame): DataFrame containing protein-protein interactions
                                    with columns 'prA' and 'prB'.

        Returns:
            Graph: NetworkX graph object representing the PPI network.
        """
        print('Creating Protein-Protein Graph')
        G_ppi = nx.from_pandas_edgelist(df_pro_pro, 'prA', 'prB')
        print(f"PPI Network: {G_ppi.number_of_nodes()} nodes, {G_ppi.number_of_edges()} edges")
        print('...done')
        return G_ppi

    def create_heterogeneous_graph(self, df_dis_pro, df_pro_pro):
        """
        Create a heterogeneous DGL graph combining protein-protein and disease-protein interactions.

        Args:
            df_dis_pro (pd.DataFrame): DataFrame containing disease-protein associations
                                    with encoded IDs ('disease_id', 'protein_id_enc').
            df_pro_pro (pd.DataFrame): DataFrame containing protein-protein interactions
                                    with encoded IDs ('src_id', 'dst_id').

        Returns:
            dgl.DGLHeteroGraph: Heterogeneous graph with node types 'disease' and 'protein',
                                and edge types 'interacts', 'associates', and 'rev_associates'.
        """
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
        """
        Convert a NetworkX PPI graph to a DGL graph and assign node features and labels.

        Args:
            ppi_graph (nx.Graph): NetworkX graph representing protein-protein interactions.
            seed_nodes (set): Set of seed node identifiers (proteins).
            feature_dim (int, optional): Dimensionality of the random feature vectors assigned
                                        to each node. Defaults to 64.

        Returns:
            dgl.DGLGraph: Homogeneous DGL graph with assigned features and
            binary labels (1 for seed nodes, 0 otherwise).
        """
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
        """
        Convert edge lists to PyTorch tensors for training and testing.

        Args:
            train_u (array-like): Source nodes of training edges.
            train_v (array-like): Target nodes of training edges.
            test_u (array-like): Source nodes of testing edges.
            test_v (array-like): Target nodes of testing edges.

        Returns:
            tuple: Tensors (train_u, train_v, test_u, test_v) as torch.long type.
        """
        train_u = torch.tensor(train_u, dtype=torch.long)
        train_v = torch.tensor(train_v, dtype=torch.long)
        test_u = torch.tensor(test_u, dtype=torch.long)
        test_v = torch.tensor(test_v, dtype=torch.long)
        return train_u, train_v, test_u, test_v

    def convert_to_heterogeneous_graph(self, G_dispro, edge_type, u, v):
        """
        Create a DGL heterogeneous graph for a specific edge type and edge list.

        Args:
            G_dispro (dgl.DGLHeteroGraph): Original heterogeneous graph containing all node types.
            edge_type (tuple): Canonical edge type (source_type, relation_type, destination_type).
            u (array-like): Source node indices for the edges.
            v (array-like): Destination node indices for the edges.

        Returns:
            dgl.DGLHeteroGraph: Heterogeneous graph containing only the specified edge type.
        """
        g = dgl.heterograph(
            {edge_type: (u, v)},
            num_nodes_dict={
                'disease': G_dispro.num_nodes('disease'),
                'protein': G_dispro.num_nodes('protein')
            }
        )
        # Find edge IDs in the original graph that match (u, v)
        u_all, v_all = G_dispro.edges(etype=edge_type)
        edge_to_id = {
            (src.item(), dst.item()): idx for idx, (src, dst) in enumerate(zip(u_all, v_all))
        }

        edge_ids = []
        mask = []  # True for found, False for not found
        for src, dst in zip(u.tolist(), v.tolist()):
            eid = edge_to_id.get((src, dst))
            if eid is not None:
                edge_ids.append(eid)
                mask.append(True)
            else:
                edge_ids.append(-1)  # Placeholder
                mask.append(False)

        edge_ids = torch.tensor(edge_ids, dtype=torch.long)
        for key in G_dispro.edges[edge_type].data.keys():
            original_feat = G_dispro.edges[edge_type].data[key]
            default_val = torch.zeros(1, dtype=original_feat.dtype, device=original_feat.device)

            new_feats = []
            for i, found in enumerate(mask):
                if found:
                    new_feats.append(original_feat[edge_ids[i]].unsqueeze(0))
                else:
                    new_feats.append(default_val.unsqueeze(0))  # Default seed score = 0.0

            g.edges[edge_type].data[key] = torch.cat(new_feats, dim=0)
        return g
