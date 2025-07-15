from typing import Dict, Tuple

import dgl
import dgl.function as fn
import torch.nn as nn
from torch import Tensor


class DotPredictor(nn.Module):
    """
    Dot Product Predictor for Link Prediction in Heterogeneous Graphs.

    Computes link scores by applying the dot product between the source and
    destination node embeddings for a given edge type. Optionally, it adds
    precomputed seed scores if available in the edge data.
    """

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_dict: Dict[str, Tensor],
        etype: Tuple[str, str, str],
        use_seed_score: bool = True
    ) -> Tensor:
        """
        Compute link prediction scores for a specified edge type using node embeddings.

        Args:
            g (dgl.DGLHeteroGraph): Input heterogeneous graph.
            h_dict (dict): Node embeddings per node type. Keys are node types,
            values are tensors (N_nodes, dim).
            etype (tuple): Edge type to compute predictions for: (src_type, relation, dst_type).
            use_seed_score (bool, optional): If True, adds 'seed_score' from
            the edge data if present. Default is True.

        Returns:
            Tensor: A 1D tensor of predicted scores for each edge of the specified edge type.
        """
        with g.local_scope():
            src_type, _, dst_type = g.to_canonical_etype(etype)
            g.nodes[src_type].data['h'] = h_dict[src_type]
            g.nodes[dst_type].data['h'] = h_dict[dst_type]

            # Compute dot product between source and destination node embeddings
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            dot_score = g.edges[etype].data['score'].squeeze()
            if use_seed_score and 'seed_score' in g.edges[etype].data:
                seed_score = g.edges[etype].data['seed_score'].squeeze()
                g.edges[etype].data['score'] = dot_score + seed_score
            else:
                g.edges[etype].data['score'] = dot_score
            return g.edges[etype].data['score']
