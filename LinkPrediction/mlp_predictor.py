from typing import Dict, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """
    Multilayer Perceptron (MLP)-based Link Predictor for Heterogeneous Graphs.

    This predictor computes link prediction scores by concatenating source and destination
    node embeddings and passing them through an MLP.
    """

    def __init__(self, in_feats: int, hidden_feats: int):
        """
        Initialize the MLP Predictor.

        Args:
            in_feats (int): Input feature size for each node embedding.
            hidden_feats (int): Hidden layer size in the MLP.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)
        )

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_dict: Dict[str, torch.Tensor],
        etype: Tuple[str, str, str],
        use_seed_score: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to compute link prediction scores for a given edge type.

        Args:
            g (dgl.DGLHeteroGraph): The input heterogeneous graph.
            h_dict (dict): Dictionary of node embeddings per node type.
            etype (tuple): Canonical edge type (src_type, relation, dst_type).
            use_seed_score (bool, optional): Whether to add the 'seed_score' (if available)
            to the final score. Defaults to False.

        Returns:
            torch.Tensor: A 1D tensor of predicted scores for each edge of the specified type.
        """
        src_type, _, dst_type = g.to_canonical_etype(etype)
        with g.local_scope():
            # Assign node embeddings to graph
            g.nodes[src_type].data['h'] = h_dict[src_type]
            g.nodes[dst_type].data['h'] = h_dict[dst_type]

            # Trigger DGL edge computation to obtain edge indices (not used directly here)
            g.apply_edges(fn.u_mul_v('h', 'h', 'dummy'), etype=etype)

            # Extract source and destination node features for current edges
            src, dst = g.edges(etype=etype)
            h_src = g.nodes[src_type].data['h'][src]
            h_dst = g.nodes[dst_type].data['h'][dst]

            # Concatenate source and destination embeddings
            h_cat = torch.cat([h_src, h_dst], dim=1)

            # Compute score via MLP
            score = self.mlp(h_cat).squeeze()

            # Optionally add seed score if available
            if use_seed_score and 'seed_score' in g.edges[etype].data:
                seed_score = g.edges[etype].data['seed_score'].squeeze()
                score += seed_score

            return score
