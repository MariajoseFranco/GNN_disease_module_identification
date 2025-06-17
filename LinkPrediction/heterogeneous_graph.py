import dgl
import torch


class HeterogeneousGraph():
    def __init__(self) -> None:
        pass

    def create_graph(self, df_dis_pro, df_pro_pro):
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
        # Build heterograph dictionary
        data_dict = {
            ('protein', 'interacts', 'protein'): (
                df_pro_pro['src_id'].values, df_pro_pro['dst_id'].values
            ),
            ('protein', 'rev_interacts', 'protein'): (
                df_pro_pro['dst_id'].values, df_pro_pro['src_id'].values
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
        return hetero_graph

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
