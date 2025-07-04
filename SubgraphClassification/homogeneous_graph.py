from typing import Optional

import dgl
import networkx as nx
import pandas as pd
import torch


class HomogeneousGraph:
    """
    Class for building and transforming protein-protein interaction (PPI) networks
    into DGL-compatible homogeneous graphs with node features.
    """

    def __init__(self) -> None:
        """Initialize an empty HomogeneousGraph instance."""
        pass

    def create_graph(self, df_pro_pro: pd.DataFrame) -> nx.Graph:
        """
        Create a NetworkX graph from protein-protein interaction data.

        Args:
            df_pro_pro (pd.DataFrame): DataFrame with columns 'prA' and 'prB'
             representing interactions.

        Returns:
            nx.Graph: Undirected NetworkX graph of the PPI network.
        """
        return nx.from_pandas_edgelist(df_pro_pro, 'prA', 'prB')

    def get_protein_pagerank(self, G_ppi: nx.Graph, top_k: Optional[int] = None) -> dict:
        """
        Compute PageRank scores for each protein in the graph.

        Args:
            G_ppi (nx.Graph): Protein-protein interaction graph.
            top_k (int, optional): Return only top_k highest ranked proteins.

        Returns:
            dict: Mapping from protein to PageRank score (or top_k highest if specified).
        """
        pr_scores = nx.pagerank(G_ppi)
        if top_k:
            return dict(sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return pr_scores

    def convert_networkx_to_dgl_graph(
        self,
        ppi_graph: nx.Graph,
        seed_scores: dict[str, float],
        node_index_mapping: dict[str, int],
        expanded_nodes: list[str],
        pr_scores: dict[str, float],
        feature_dim: int = 64
    ) -> dgl.DGLGraph:
        """
        Convert a NetworkX graph to a DGL graph with node features.

        Args:
            ppi_graph (nx.Graph): NetworkX graph of the PPI network.
            seed_scores (dict[str, float]): Seed score values for known proteins.
            node_index_mapping (dict[str, int]): Mapping from protein names to node indices.
            expanded_nodes (list[str]): List of proteins in the subgraph.
            pr_scores (dict[str, float]): PageRank scores for each node.
            feature_dim (int, optional): Dimensionality of random node features.

        Returns:
            dgl.DGLGraph: DGL graph with combined node features.
        """
        dgl_graph = self.convert_to_dgl_graph(ppi_graph)
        num_nodes = dgl_graph.num_nodes()
        node_index_mapping = {k: v for k, v in node_index_mapping.items() if v < num_nodes}

        rand_features = self.generate_random_features(num_nodes, feature_dim)
        score_features = self.generate_seed_scores_features(
            seed_scores, node_index_mapping, num_nodes
        )
        degree_features = self.generate_degree_features(ppi_graph, expanded_nodes)
        pagerank_features = self.generate_pagerank_features(pr_scores, expanded_nodes)

        combined_features = self.combine_features(
            rand_features, score_features, degree_features, pagerank_features
        )
        dgl_graph.ndata['feat'] = combined_features

        return dgl_graph

    def convert_to_dgl_graph(self, ppi_graph: nx.Graph) -> dgl.DGLGraph:
        """
        Convert a NetworkX graph to a DGL graph.

        Args:
            ppi_graph (nx.Graph): Input graph.

        Returns:
            dgl.DGLGraph: Converted DGL graph.
        """
        return dgl.from_networkx(ppi_graph)

    def generate_random_features(self, num_nodes: int, feature_dim: int) -> torch.Tensor:
        """
        Generate random feature vectors for nodes.

        Args:
            num_nodes (int): Number of nodes.
            feature_dim (int): Feature dimensionality.

        Returns:
            torch.Tensor: Random features of shape (num_nodes, feature_dim).
        """
        return torch.randn((num_nodes, feature_dim))

    def generate_seed_scores_features(
        self,
        seed_scores: dict[str, float],
        node_index_mapping: dict[str, int],
        num_nodes: int
    ) -> torch.Tensor:
        """
        Generate node features based on known seed scores.

        Args:
            seed_scores (dict): Mapping from protein to score.
            node_index_mapping (dict): Mapping from protein to index.
            num_nodes (int): Total number of nodes.

        Returns:
            torch.Tensor: Feature tensor of shape (num_nodes, 1).
        """
        scores = torch.zeros(num_nodes, dtype=torch.float32)
        for node, score in seed_scores.items():
            if node in node_index_mapping:
                scores[node_index_mapping[node]] = score
        return scores.unsqueeze(1)

    def generate_degree_features(
            self, ppi_graph: nx.Graph, expanded_nodes: list[str]
    ) -> torch.Tensor:
        """
        Compute degree-based node features.

        Args:
            ppi_graph (nx.Graph): Original PPI graph.
            expanded_nodes (list): List of nodes in the subgraph.

        Returns:
            torch.Tensor: Degree features of shape (len(expanded_nodes), 1).
        """
        degree_features = torch.tensor(
            [ppi_graph.degree(n) for n in expanded_nodes], dtype=torch.float32
        )
        return degree_features.unsqueeze(1)

    def generate_pagerank_features(
            self, pr_scores: dict[str, float], expanded_nodes: list[str]
    ) -> torch.Tensor:
        """
        Generate node features based on PageRank scores.

        Args:
            pr_scores (dict): PageRank scores from the full graph.
            expanded_nodes (list): Nodes to extract scores for.

        Returns:
            torch.Tensor: PageRank features of shape (len(expanded_nodes), 1).
        """
        pagerank_features = torch.tensor(
            [pr_scores.get(n, 0.0) for n in expanded_nodes], dtype=torch.float32
        )
        return pagerank_features.unsqueeze(1)

    def combine_features(
        self,
        rand_features: torch.Tensor,
        score_features: torch.Tensor,
        degree_features: torch.Tensor,
        pagerank_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate all node features into a single tensor.

        Args:
            rand_features (torch.Tensor): Randomly initialized features.
            score_features (torch.Tensor): Seed score features.
            degree_features (torch.Tensor): Degree features.
            pagerank_features (torch.Tensor): PageRank features.

        Returns:
            torch.Tensor: Combined features of shape (N, D).
        """
        return torch.cat([rand_features, score_features, degree_features, pagerank_features], dim=1)

    def get_node_scoring(
            self, disease_pro_mapping: dict[str, dict[str, float]], disease: str
    ) -> dict[str, float]:
        """
        Retrieves the protein scoring dictionary for a specific disease.

        Args:
            disease_pro_mapping (dict): Dictionary mapping diseases to protein–score dictionaries.
            disease (str): The disease of interest.

        Returns:
            dict: Dictionary mapping protein IDs to scores for the specified disease.
        """
        return disease_pro_mapping[disease]

    def get_seed_nodes(self, G_ppi: nx.Graph, node_scoring: dict[str, float]) -> set[str]:
        """
        Identifies seed proteins for a disease that are present in the PPI graph.

        Args:
            G_ppi (nx.Graph): Protein–protein interaction graph.
            node_scoring (dict): Dictionary of proteins with associated scores.

        Returns:
            set: Seed proteins that exist in the PPI graph.
        """
        seed_nodes_complete = {key for key, _ in node_scoring.items()}
        seed_nodes = seed_nodes_complete.intersection(G_ppi.nodes())
        return seed_nodes

    def get_expanded_nodes(
            self, seed_nodes: set[str], all_proteins: set[str], pr_scores: dict[str, float]
    ) -> list[str]:
        """
        Generates an expanded set of nodes including seed proteins and sampled negatives.

        Args:
            seed_nodes (set): Known disease-associated proteins.
            all_proteins (set): All proteins in the PPI network.
            pr_scores (dict): Precomputed PageRank scores for each protein.

        Returns:
            list: List of expanded nodes including seed and negative candidates.
        """
        known_proteins = seed_nodes
        candidates = [p for p in all_proteins if p not in known_proteins]
        candidates = sorted(candidates, key=lambda p: pr_scores.get(p, 0), reverse=True)
        sampled_negatives = candidates[:min(10 * len(known_proteins), len(candidates))]

        expanded_nodes = list(seed_nodes) + sampled_negatives
        return expanded_nodes

    def filtering_info(
        self,
        G_ppi: nx.Graph,
        expanded_nodes: list[str],
        node_scoring: dict[str, float]
    ) -> tuple[nx.Graph, list[str], dict[str, float], dict[str, int]]:
        """
        Filters the expanded subgraph by removing isolated nodes and builds necessary mappings.

        Args:
            G_ppi (nx.Graph): Full protein–protein interaction graph.
            expanded_nodes (list): Nodes selected for subgraph extraction.
            node_scoring (dict): Dictionary of protein scores.

        Returns:
            tuple:
                - nx_subgraph (nx.Graph): Filtered subgraph.
                - non_isolated_nodes (list): Nodes with degree > 0.
                - node_scoring_filtered (dict): Scores for non-isolated nodes.
                - node_index_filtered (dict): Mapping from protein ID to integer index.
        """
        nx_subgraph = G_ppi.subgraph(expanded_nodes).copy()
        non_isolated_nodes = [n for n in nx_subgraph.nodes() if nx_subgraph.degree(n) > 0]

        node_scoring_filtered = {k: node_scoring.get(k, 0.0) for k in non_isolated_nodes}
        node_index_filtered = {k: i for i, k in enumerate(non_isolated_nodes)}
        return nx_subgraph, non_isolated_nodes, node_scoring_filtered, node_index_filtered
