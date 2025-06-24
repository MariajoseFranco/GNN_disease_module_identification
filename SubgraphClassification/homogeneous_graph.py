import dgl
import networkx as nx
import pandas as pd
import torch


class HomogeneousGraph():
    def __init__(self) -> None:
        pass

    def create_graph(self, df_pro_pro: pd.DataFrame) -> nx.Graph:
        """
        Create a homogeneous graph from protein-protein interaction data.

        Args:
            df_pro_pro (pd.DataFrame): DataFrame containing protein-protein interactions
                                    with columns 'prA' and 'prB'.

        Returns:
            Graph: NetworkX graph object representing the PPI network.
        """
        G_ppi = nx.from_pandas_edgelist(df_pro_pro, 'prA', 'prB')
        return G_ppi

    def get_protein_pagerank(self, G_ppi, top_k=None):
        pr_scores = nx.pagerank(G_ppi)
        if top_k:
            return sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return pr_scores

    def convert_networkx_to_dgl_graph(
        self,
        ppi_graph: nx.Graph,
        seed_scores: dict[str, float],
        node_index_mapping: dict[str, int],
        expanded_nodes,
        pr_scores,
        feature_dim: int = 64
    ) -> tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return dgl.from_networkx(ppi_graph)

    def generate_random_features(self, num_nodes: int, feature_dim: int) -> torch.Tensor:
        return torch.randn((num_nodes, feature_dim))

    def generate_seed_scores_features(
            self, seed_scores: dict[str, float], node_index_mapping: dict[str, int], num_nodes: int
    ) -> torch.Tensor:
        scores = torch.zeros(num_nodes, dtype=torch.float32)
        for node, score in seed_scores.items():
            if node in node_index_mapping:
                scores[node_index_mapping[node]] = score
        return scores.unsqueeze(1)

    def generate_degree_features(self, ppi_graph, expanded_nodes):
        degree_features = torch.tensor(
            [ppi_graph.degree(n) for n in expanded_nodes], dtype=torch.float32
        )
        return degree_features.unsqueeze(1)

    def generate_pagerank_features(self, pr_scores, expanded_nodes):
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
        return torch.cat([rand_features, score_features, degree_features, pagerank_features], dim=1)
