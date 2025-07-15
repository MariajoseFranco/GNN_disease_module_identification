from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import dgl
import numpy as np
import pandas as pd
import torch

from LinkPrediction.utils.helper_functions import \
    obtaining_removing_reverse_edges


class HeterogeneousGraph:
    """
    Class for constructing heterogeneous DGL graphs representing disease–protein
    and protein–protein interactions.
    """

    def __init__(self) -> None:
        """
        Initialize the HeterogeneousGraph class.
        """
        pass

    def create_graph(
        self,
        df_dis_pro: pd.DataFrame,
        df_pro_pro: pd.DataFrame
    ) -> dgl.DGLHeteroGraph:
        """
        Create a heterogeneous DGL graph combining disease–protein and
        protein–protein interactions.

        Node types:
            - 'disease'
            - 'protein'

        Edge types:
            - ('protein', 'interacts', 'protein')
            - ('protein', 'rev_interacts', 'protein')
            - ('disease', 'associates', 'protein')
            - ('protein', 'rev_associates', 'disease')

        Args:
            df_dis_pro (pd.DataFrame): DataFrame containing disease–protein associations.
                Expected columns: 'disease_id', 'protein_id_enc'.
            df_pro_pro (pd.DataFrame): DataFrame containing protein–protein interactions.
                Expected columns: 'src_id', 'dst_id'.

        Returns:
            dgl.DGLHeteroGraph: The constructed heterogeneous graph with both
            forward and reverse edges for disease–protein and protein–protein relations.
        """
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

        hetero_graph = dgl.heterograph(data_dict)
        return hetero_graph

    def create_multitype_graph_with_reverse_edges(
        self,
        df_ddi_dru: pd.DataFrame,
        df_ddi_phe: pd.DataFrame,
        df_dis_dru_the: pd.DataFrame,
        df_dis_pat: pd.DataFrame,
        df_dis_pro: pd.DataFrame,
        df_dis_sym: pd.DataFrame,
        df_dru_dru: pd.DataFrame,
        df_dru_pro: pd.DataFrame,
        df_dru_sym_ind: pd.DataFrame,
        df_dru_sym_sef: pd.DataFrame,
        df_pro_pat: pd.DataFrame,
        df_pro_pro: pd.DataFrame,
        add_reverse: bool = True
    ) -> dgl.DGLHeteroGraph:
        """
        Constructs a heterogeneous DGL graph from multiple biomedical relations
        between drugs, diseases, proteins, phenotypes, and pathways.

        Optionally, reverse edges are added for each relation.

        Edge types created include (but are not limited to):
            - ('drug-drug-interaction', 'ddi_dru', 'drug')
            - ('disease', 'dis_dru_the', 'drug')
            - ('disease', 'dis_pro', 'protein')
            - ('protein', 'proA_proB', 'protein')
            - Reverse edges prefixed with 'rev_' if add_reverse=True.

        Args:
            df_ddi_dru (pd.DataFrame): Drug–drug interactions (columns: 'ddi', 'dru').
            df_ddi_phe (pd.DataFrame): Drug–phenotype associations (columns: 'ddi', 'phe').
            df_dis_dru_the (pd.DataFrame): Disease–drug therapeutic associations
            (columns: 'dis', 'dru').
            df_dis_pat (pd.DataFrame): Disease–pathway associations (columns: 'dis', 'pat').
            df_dis_pro (pd.DataFrame): Disease–protein associations (columns: 'dis', 'pro').
            df_dis_sym (pd.DataFrame): Disease–phenotype associations (columns: 'dis', 'sym').
            df_dru_dru (pd.DataFrame): Drug–drug interactions (columns: 'drA', 'drB').
            df_dru_pro (pd.DataFrame): Drug–protein associations (columns: 'dru', 'pro').
            df_dru_sym_ind (pd.DataFrame): Drug–phenotype indications (columns: 'dru', 'sym').
            df_dru_sym_sef (pd.DataFrame): Drug–phenotype side effects (columns: 'dru', 'sym').
            df_pro_pat (pd.DataFrame): Protein–pathway associations (columns: 'pro', 'pat').
            df_pro_pro (pd.DataFrame): Protein–protein interactions (columns: 'prA', 'prB').
            add_reverse (bool, optional): Whether to add reverse edges for each relation.
                Reverse edges are prefixed with 'rev_'. Defaults to True.

        Returns:
            dgl.DGLHeteroGraph: A heterogeneous graph with multiple node and edge types.
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

    def create_heterograph_with_mapped_ids(
        self,
        edge_specs: List[Tuple[str, str, str, np.ndarray, np.ndarray]],
        add_reverse: bool = True
    ) -> Tuple[dgl.DGLHeteroGraph, Dict[str, Dict[Any, int]], Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]]]:
        """
        Create a DGL heterogeneous graph with automatically mapped node IDs.

        Each unique node value is mapped to an integer ID per node type to create compact ID spaces.
        Optionally, reverse edges are added for each relation.

        Args:
            edge_specs (List[Tuple[str, str, str, np.ndarray, np.ndarray]]):
                List of edge definitions where each element is a tuple:
                (source_node_type, relation_type, destination_node_type,
                source_ids, destination_ids).
            add_reverse (bool, optional): Whether to add reverse edges with 'rev_' prefix.
            Defaults to True.

        Returns:
            Tuple containing:
                - hetero_graph (dgl.DGLHeteroGraph): The constructed heterogeneous graph.
                - node_maps (Dict[str, Dict[Any, int]]): Mapping from original node IDs to
                compact integer IDs per node type.
                - full_edges (Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]]):
                    The final dictionary of edges with mapped integer node IDs.
        """
        node_type_to_values = defaultdict(set)

        # Collect all unique node IDs per node type
        for src_type, _, dst_type, src_array, dst_array in edge_specs:
            node_type_to_values[src_type].update(src_array)
            node_type_to_values[dst_type].update(dst_array)

        # Map original node IDs to integer indices
        node_maps = {
            node_type: {val: idx for idx, val in enumerate(sorted(values))}
            for node_type, values in node_type_to_values.items()
        }

        # Build edges with mapped node IDs
        edges_dict = {}
        for src_type, rel_type, dst_type, src_array, dst_array in edge_specs:
            src_ids = pd.Series(src_array).map(node_maps[src_type]).values.astype(np.int64)
            dst_ids = pd.Series(dst_array).map(node_maps[dst_type]).values.astype(np.int64)
            edges_dict[(src_type, rel_type, dst_type)] = (src_ids, dst_ids)

        # Optionally add reverse edges
        full_edges = dict(edges_dict)
        if add_reverse:
            for (src_type, rel_type, dst_type), (src_ids, dst_ids) in edges_dict.items():
                rev_rel_type = f'rev_{rel_type}'
                full_edges[(dst_type, rev_rel_type, src_type)] = (dst_ids, src_ids)

        hetero_graph = dgl.heterograph(full_edges)
        return hetero_graph, node_maps, full_edges

    def convert_to_heterogeneous_graph(
        self,
        G_dispro: dgl.DGLHeteroGraph,
        edge_type: Tuple[str, str, str],
        u: Union[torch.Tensor, list],
        v: Union[torch.Tensor, list]
    ) -> dgl.DGLHeteroGraph:
        """
        Create a new DGL heterogeneous graph containing only the specified edge type and edge list.
        This function also copies relevant edge features from the original graph.

        Args:
            G_dispro (dgl.DGLHeteroGraph): The original full heterogeneous graph.
            edge_type (Tuple[str, str, str]): Canonical edge type
            (src_type, relation_type, dst_type).
            u (Tensor or list): Source node indices for the selected edges.
            v (Tensor or list): Destination node indices for the selected edges.

        Returns:
            dgl.DGLHeteroGraph: A new heterogeneous graph with only the specified
            edge type and its features.
        """
        src_type, _, dst_type = edge_type

        # Define node counts for the new graph
        num_nodes_dict = {
            src_type: G_dispro.num_nodes(src_type),
            dst_type: G_dispro.num_nodes(dst_type)
        }

        # Build new heterograph with the specified edges
        g = dgl.heterograph({edge_type: (u, v)}, num_nodes_dict=num_nodes_dict)

        # Map edge pairs to original edge IDs
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
                edge_ids.append(-1)  # Use default value for missing edges
                mask.append(False)

        edge_ids = torch.tensor(edge_ids, dtype=torch.long)

        # Copy edge features from the original graph to the new graph
        for key in G_dispro.edges[edge_type].data.keys():
            original_feat = G_dispro.edges[edge_type].data[key]
            default_val = torch.zeros(
                original_feat.shape[1:], dtype=original_feat.dtype, device=original_feat.device
            )

            new_feats = []
            for i, found in enumerate(mask):
                if found:
                    new_feats.append(original_feat[edge_ids[i]].unsqueeze(0))
                else:
                    new_feats.append(default_val.unsqueeze(0))
            g.edges[edge_type].data[key] = torch.cat(new_feats, dim=0)

        return g

    def obtaining_edges_and_score_edges(
        self,
        G_dispro: dgl.DGLHeteroGraph,
        seed_edge_scores: Dict[Tuple[int, int], float],
        edge_type: Tuple[str, str, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the edges of a specific edge type from a heterogeneous
        graph and assign seed scores.

        This function returns:
            - The source and destination node indices of the edges of the specified type.
            - A tensor of seed scores aligned with the edges.

        Args:
            G_dispro (dgl.DGLHeteroGraph): The heterogeneous graph containing edges and features.
            seed_edge_scores (Dict[Tuple[int, int], float]): Dictionary mapping
            (src, dst) pairs to seed scores.
            edge_type (Tuple[str, str, str]): The specific edge type to process
            (src_type, relation, dst_type).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - u (Tensor): Source node indices of the edges.
                - v (Tensor): Destination node indices of the edges.
                - seed_score_tensor (Tensor): Seed scores for each edge of shape (num_edges, 1).
        """
        u, v = G_dispro.edges(etype=edge_type)
        num_edges = G_dispro.num_edges(etype=edge_type)
        seed_score_tensor = torch.zeros(num_edges, 1)

        _, relation_edge_type, _ = edge_type

        for i, (src, dst) in enumerate(zip(u.tolist(), v.tolist())):
            if relation_edge_type == 'associates':
                key = (src, dst)  # disease → protein
            elif relation_edge_type == 'rev_associates':
                key = (dst, src)  # protein → disease (reverse association)
            else:
                key = None  # For other edge types, no seed score is assigned

            if key is not None and key in seed_edge_scores:
                seed_score_tensor[i] = seed_edge_scores[key]
        return u, v, seed_score_tensor

    def assigining_seed_score_to_edges(
        self,
        G_dispro: dgl.DGLHeteroGraph,
        seed_score_tensor: torch.Tensor,
        edge_type: Tuple[str, str, str],
        device: torch.device
    ) -> dgl.DGLHeteroGraph:
        """
        Assigns precomputed seed scores to the edges of a specific edge type
        in the heterogeneous graph.

        The seed scores are typically used to inject prior knowledge
        (e.g., known disease–protein associations) into the model by
        influencing the prediction scores.

        Args:
            G_dispro (dgl.DGLHeteroGraph): The heterogeneous graph to modify.
            seed_score_tensor (torch.Tensor): Tensor of shape (num_edges, 1)
            containing the seed scores.
            edge_type (Tuple[str, str, str]): Canonical edge type
            (source_type, relation, destination_type).
            device (torch.device): The target device to move the tensor to (CPU or GPU).

        Returns:
            dgl.DGLHeteroGraph: The graph with 'seed_score' added to the specified edge type.
        """
        G_dispro.edges[edge_type].data['seed_score'] = seed_score_tensor.to(device)
        return G_dispro

    def assigining_nodes_features(
        self,
        G_dispro: dgl.DGLHeteroGraph,
        feat_dim: int,
        device: torch.device
    ) -> Tuple[dgl.DGLHeteroGraph, Dict[str, torch.Tensor]]:
        """
        Assigns random feature vectors to 'disease' and 'protein' nodes in the heterogeneous graph.

        Each node receives a feature vector sampled from a standard normal distribution.
        The features are commonly used as input to graph neural networks when explicit biological
        features are not provided.

        Args:
            G_dispro (dgl.DGLHeteroGraph): The heterogeneous graph to modify.
            feat_dim (int): The dimensionality of the feature vectors to generate.
            device (torch.device): The target device to move the features to (CPU or GPU).

        Returns:
            Tuple:
                - dgl.DGLHeteroGraph: The graph with node features added.
                - Dict[str, torch.Tensor]: Dictionary mapping node types to their feature tensors.
        """
        G_dispro.nodes['disease'].data['feat'] = torch.randn(
            G_dispro.num_nodes('disease'), feat_dim
        )
        G_dispro.nodes['protein'].data['feat'] = torch.randn(
            G_dispro.num_nodes('protein'), feat_dim
        )
        features = {
            'disease': G_dispro.nodes['disease'].data['feat'].to(device),
            'protein': G_dispro.nodes['protein'].data['feat'].to(device)
        }
        return G_dispro, features

    def assigining_nodes_features_full_graph(
        self,
        G_dispro: dgl.DGLHeteroGraph,
        feat_dim: int,
        device: torch.device
    ) -> Tuple[dgl.DGLHeteroGraph, Dict[str, torch.Tensor]]:
        """
        Assigns random feature vectors to nodes in the heterogeneous graph.

        Each node receives a feature vector sampled from a standard normal distribution.
        The features are commonly used as input to graph neural networks when explicit biological
        features are not provided.

        Args:
            G_dispro (dgl.DGLHeteroGraph): The heterogeneous graph to modify.
            feat_dim (int): The dimensionality of the feature vectors to generate.
            device (torch.device): The target device to move the features to (CPU or GPU).

        Returns:
            Tuple:
                - dgl.DGLHeteroGraph: The graph with node features added.
                - Dict[str, torch.Tensor]: Dictionary mapping node types to their feature tensors.
        """
        G_dispro.nodes['disease'].data['feat'] = torch.randn(
            G_dispro.num_nodes('disease'), feat_dim
        )
        G_dispro.nodes['protein'].data['feat'] = torch.randn(
            G_dispro.num_nodes('protein'), feat_dim
        )
        G_dispro.nodes['drug'].data['feat'] = torch.randn(
            G_dispro.num_nodes('drug'), feat_dim
        )
        G_dispro.nodes['pathway'].data['feat'] = torch.randn(
            G_dispro.num_nodes('pathway'), feat_dim
        )
        G_dispro.nodes['phenotype'].data['feat'] = torch.randn(
            G_dispro.num_nodes('phenotype'), feat_dim
        )

        features = {
            'disease': G_dispro.nodes['disease'].data['feat'].to(device),
            'protein': G_dispro.nodes['protein'].data['feat'].to(device),
            'drug': G_dispro.nodes['drug'].data['feat'].to(device),
            'pathway': G_dispro.nodes['pathway'].data['feat'].to(device),
            'phenotype': G_dispro.nodes['phenotype'].data['feat'].to(device)
        }
        return G_dispro, features

    def removing_edges_from_graph(
        self,
        G_dispro: dgl.DGLHeteroGraph,
        val_pos_u: torch.Tensor,
        val_pos_v: torch.Tensor,
        test_pos_u: torch.Tensor,
        test_pos_v: torch.Tensor,
        rev_u: torch.Tensor,
        rev_v: torch.Tensor,
        val_eids: np.ndarray,
        test_eids: np.ndarray,
        edge_type: Tuple[str, str, str],
        rev_edge_type: Tuple[str, str, str]
    ) -> dgl.DGLHeteroGraph:
        """
        Removes validation and test positive edges, as well as their corresponding reverse edges,
        from a heterogeneous graph. This is done to prevent information leakage during
        link prediction tasks.

        Args:
            G_dispro (dgl.DGLHeteroGraph): The original heterogeneous graph.
            val_pos_u (torch.Tensor): Source nodes of validation positive edges.
            val_pos_v (torch.Tensor): Destination nodes of validation positive edges.
            test_pos_u (torch.Tensor): Source nodes of test positive edges.
            test_pos_v (torch.Tensor): Destination nodes of test positive edges.
            rev_u (torch.Tensor): Source nodes of reverse edges in the reverse edge type.
            rev_v (torch.Tensor): Destination nodes of reverse edges in the reverse edge type.
            val_eids (np.ndarray): Edge IDs of validation edges to remove (forward direction).
            test_eids (np.ndarray): Edge IDs of test edges to remove (forward direction).
            edge_type (tuple): Canonical forward edge type (src_type, relation, dst_type).
            rev_edge_type (tuple): Canonical reverse edge type (dst_type, rev_relation, src_type).

        Returns:
            dgl.DGLHeteroGraph: The graph with validation/test edges and their
            reverse edges removed.
        """
        rev_eids_to_remove = obtaining_removing_reverse_edges(
            val_pos_u, val_pos_v, test_pos_u, test_pos_v, rev_u, rev_v
        )

        # Remove forward edges (val/test)
        G_tmp = dgl.remove_edges(G_dispro, np.concatenate([val_eids, test_eids]), etype=edge_type)

        # Remove reverse edges if any were found
        if len(rev_eids_to_remove) > 0:
            train_g = dgl.remove_edges(G_tmp, rev_eids_to_remove, etype=rev_edge_type)
        else:
            train_g = G_tmp

        return train_g
