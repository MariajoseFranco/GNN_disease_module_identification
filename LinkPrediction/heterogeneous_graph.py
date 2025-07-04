from collections import defaultdict

import dgl
import numpy as np
import pandas as pd
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

    def create_multitype_graph_with_reverse_edges(
            self,
            df_ddi_dru, df_ddi_phe, df_dis_dru_the,
            df_dis_pat, df_dis_pro, df_dis_sym,
            df_dru_dru, df_dru_pro, df_dru_sym_ind,
            df_dru_sym_sef, df_pro_pat, df_pro_pro, add_reverse=True
    ):
        """
        Create a DGL heterogeneous graph, optionally adding reverse edges.

        Args:
            edges_dict (dict): Dictionary with keys as edge type tuples
                            (src_type, relation_type, dst_type) and values as (src_ids, dst_ids).
            add_reverse (bool): Whether to automatically add reverse edges with 'rev_' prefix.

        Returns:
            dgl.DGLHeteroGraph: The heterogeneous graph with optional reverse edges.
        """
        edges_dict = {
            ('drug-drug-interaction', 'ddi_dru', 'drug'): (
                df_ddi_dru['ddi'].values, df_ddi_dru['dru'].values
            ),
            ('drug-drug-interaction', 'ddi_phe', 'phenotype'): (
                df_ddi_phe['ddi'].values, df_ddi_phe['phe'].values
            ),
            ('disease', 'dis_dru_the', 'drug'): (
                df_dis_dru_the['dis'].values, df_dis_dru_the['dru'].values
            ),
            ('disease', 'dis_pat', 'pathway'): (
                df_dis_pat['dis'].values, df_dis_pat['pat'].values
            ),
            ('disease', 'dis_pro', 'protein'): (
                df_dis_pro['dis'].values, df_dis_pro['pro'].values
            ),
            ('disease', 'dse_sym', 'phenotype'): (
                df_dis_sym['dis'].values, df_dis_sym['sym'].values
            ),
            ('drug', 'druA_druB', 'drug'): (
                df_dru_dru['drA'].values, df_dru_dru['drB'].values
            ),
            ('drug', 'dru_pro', 'protein'): (
                df_dru_pro['dru'].values, df_dru_pro['pro'].values
            ),
            ('drug', 'dru_sym_ind', 'phenotype'): (
                df_dru_sym_ind['dru'].values, df_dru_sym_ind['sym'].values
            ),
            ('drug', 'dru_sym_sef', 'phenotype'): (
                df_dru_sym_sef['dru'].values, df_dru_sym_sef['sym'].values
            ),
            ('protein', 'pro_pat', 'pathway'): (
                df_pro_pat['pro'].values, df_pro_pat['pat'].values
            ),
            ('protein', 'proA_proB', 'protein'): (
                df_pro_pro['prA'].values, df_pro_pro['prB'].values
            ),
        }
        full_edges = dict(edges_dict)

        if add_reverse:
            for (src_type, rel_type, dst_type), (src_ids, dst_ids) in edges_dict.items():
                rev_rel_type = f'rev_{rel_type}'
                rev_edge_type = (dst_type, rev_rel_type, src_type)
                full_edges[rev_edge_type] = (dst_ids, src_ids)

        for k, (src, dst) in edges_dict.items():
            print(f"{k}: src type = {src.dtype}, dst type = {dst.dtype}")

        hetero_graph = dgl.heterograph(full_edges)
        return hetero_graph

    def create_heterograph_with_mapped_ids(self, edge_specs, add_reverse=True):
        """
        Crea un grafo heterogéneo de DGL a partir de múltiples relaciones y mapea automáticamente los nodos.

        Args:
            edge_specs (list of tuples): Cada elemento debe ser de la forma:
                (src_type, relation_type, dst_type, src_array, dst_array)
            add_reverse (bool): Si se añaden relaciones inversas automáticamente.

        Returns:
            hetero_graph (dgl.DGLHeteroGraph): El grafo heterogéneo con IDs numéricos.
            node_maps (dict): Diccionario que mapea nodos originales a índices por tipo.
            full_edges (dict): Diccionario con tuplas de tipo de arista → (src_ids, dst_ids)
        """
        # Paso 1: recolectar todos los nodos únicos por tipo
        node_type_to_values = defaultdict(set)
        for src_type, _, dst_type, src_array, dst_array in edge_specs:
            node_type_to_values[src_type].update(src_array)
            node_type_to_values[dst_type].update(dst_array)

        # Paso 2: asignar índices únicos por tipo
        node_maps = {
            node_type: {val: idx for idx, val in enumerate(sorted(values))}
            for node_type, values in node_type_to_values.items()
        }

        # Paso 3: construir diccionario de aristas con IDs numéricos
        edges_dict = {}
        for src_type, rel_type, dst_type, src_array, dst_array in edge_specs:
            src_ids = pd.Series(src_array).map(node_maps[src_type]).values.astype(np.int64)
            dst_ids = pd.Series(dst_array).map(node_maps[dst_type]).values.astype(np.int64)
            edges_dict[(src_type, rel_type, dst_type)] = (src_ids, dst_ids)

        # Paso 4: añadir aristas reversas si se requiere
        full_edges = dict(edges_dict)
        if add_reverse:
            for (src_type, rel_type, dst_type), (src_ids, dst_ids) in edges_dict.items():
                rev_rel_type = f'rev_{rel_type}'
                full_edges[(dst_type, rev_rel_type, src_type)] = (dst_ids, src_ids)

        # Paso 5: crear grafo DGL
        hetero_graph = dgl.heterograph(full_edges)
        return hetero_graph, node_maps, full_edges

    def convert_to_heterogeneous_graph(self, G_dispro, edge_type, u, v):
        """
        Create a DGL heterogeneous graph for a specific edge type and edge list.

        Args:
            G_dispro (dgl.DGLHeteroGraph): Original heterogeneous graph.
            edge_type (tuple): Canonical edge type (source_type, relation_type, destination_type).
            u (array-like or Tensor): Source node indices.
            v (array-like or Tensor): Destination node indices.

        Returns:
            dgl.DGLHeteroGraph: Heterogeneous graph with only the specified edge type.
        """
        # Detect node types from edge_type
        src_type, _, dst_type = edge_type

        # Build node count dict only for involved node types
        num_nodes_dict = {
            src_type: G_dispro.num_nodes(src_type),
            dst_type: G_dispro.num_nodes(dst_type)
        }

        # Create the heterograph with just the edge_type
        g = dgl.heterograph({edge_type: (u, v)}, num_nodes_dict=num_nodes_dict)

        # Get all edges of that type in the original graph
        u_all, v_all = G_dispro.edges(etype=edge_type)
        edge_to_id = {
            (src.item(), dst.item()): idx for idx, (src, dst) in enumerate(zip(u_all, v_all))
        }

        edge_ids = []
        mask = []
        for src, dst in zip(u.tolist(), v.tolist()):
            eid = edge_to_id.get((src, dst))
            if eid is not None:
                edge_ids.append(eid)
                mask.append(True)
            else:
                edge_ids.append(-1)  # Will be replaced with default
                mask.append(False)

        edge_ids = torch.tensor(edge_ids, dtype=torch.long)

        # Copy edge features from original graph (if they exist)
        for key in G_dispro.edges[edge_type].data.keys():
            original_feat = G_dispro.edges[edge_type].data[key]
            default_val = torch.zeros(original_feat.shape[1:], dtype=original_feat.dtype, device=original_feat.device)

            new_feats = []
            for i, found in enumerate(mask):
                if found:
                    new_feats.append(original_feat[edge_ids[i]].unsqueeze(0))
                else:
                    new_feats.append(default_val.unsqueeze(0))
            g.edges[edge_type].data[key] = torch.cat(new_feats, dim=0)

        return g

    # def convert_to_heterogeneous_graph(self, G_dispro, edge_type, u, v):
    #     """
    #     Create a DGL heterogeneous graph for a specific edge type and edge list.

    #     Args:
    #         G_dispro (dgl.DGLHeteroGraph): Original heterogeneous graph containing all node types.
    #         edge_type (tuple): Canonical edge type (source_type, relation_type, destination_type).
    #         u (array-like): Source node indices for the edges.
    #         v (array-like): Destination node indices for the edges.

    #     Returns:
    #         dgl.DGLHeteroGraph: Heterogeneous graph containing only the specified edge type.
    #     """
    #     g = dgl.heterograph(
    #         {edge_type: (u, v)},
    #         num_nodes_dict={
    #             'disease': G_dispro.num_nodes('disease'),
    #             'protein': G_dispro.num_nodes('protein')
    #         }
    #     )
    #     # Find edge IDs in the original graph that match (u, v)
    #     u_all, v_all = G_dispro.edges(etype=edge_type)
    #     edge_to_id = {
    #         (src.item(), dst.item()): idx for idx, (src, dst) in enumerate(zip(u_all, v_all))
    #     }

    #     edge_ids = []
    #     mask = []  # True for found, False for not found
    #     for src, dst in zip(u.tolist(), v.tolist()):
    #         eid = edge_to_id.get((src, dst))
    #         if eid is not None:
    #             edge_ids.append(eid)
    #             mask.append(True)
    #         else:
    #             edge_ids.append(-1)  # Placeholder
    #             mask.append(False)

    #     edge_ids = torch.tensor(edge_ids, dtype=torch.long)
    #     for key in G_dispro.edges[edge_type].data.keys():
    #         original_feat = G_dispro.edges[edge_type].data[key]
    #         default_val = torch.zeros(1, dtype=original_feat.dtype, device=original_feat.device)

    #         new_feats = []
    #         for i, found in enumerate(mask):
    #             if found:
    #                 new_feats.append(original_feat[edge_ids[i]].unsqueeze(0))
    #             else:
    #                 new_feats.append(default_val.unsqueeze(0))  # Default seed score = 0.0

    #         g.edges[edge_type].data[key] = torch.cat(new_feats, dim=0)
    #     return g
