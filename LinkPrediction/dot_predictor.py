import dgl.function as fn
import torch.nn as nn


class DotPredictor(nn.Module):
    def forward(self, g, h_dict, etype, use_seed_score=True):
        """
        Compute link prediction scores for a heterogeneous graph using a dot
        product between node embeddings.

        Args:
            g (dgl.DGLHeteroGraph): The heterogeneous graph.
            h_dict (dict): Dictionary of node embeddings, with node types as keys.
            etype (tuple): Edge type for which predictions are computed
            (src_type, relation, dst_type).

        Returns:
            torch.Tensor: A 1D tensor of scores for each edge of the specified edge type.
        """
        with g.local_scope():
            src_type, _, dst_type = g.to_canonical_etype(etype)
            g.nodes[src_type].data['h'] = h_dict[src_type]
            g.nodes[dst_type].data['h'] = h_dict[dst_type]

            # Compute dot product
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            dot_score = g.edges[etype].data['score'].squeeze()
            if use_seed_score and 'seed_score' in g.edges[etype].data:
                seed_score = g.edges[etype].data['seed_score'].squeeze()
                g.edges[etype].data['score'] = dot_score + seed_score
            else:
                g.edges[etype].data['score'] = dot_score
            return g.edges[etype].data['score']
