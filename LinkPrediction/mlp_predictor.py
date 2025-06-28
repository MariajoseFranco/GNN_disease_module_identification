import dgl.function as fn
import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)
        )

    def forward(self, g, h_dict, etype, use_seed_score=False):
        src_type, _, dst_type = g.to_canonical_etype(etype)
        with g.local_scope():
            g.nodes[src_type].data['h'] = h_dict[src_type]
            g.nodes[dst_type].data['h'] = h_dict[dst_type]

            g.apply_edges(fn.u_mul_v('h', 'h', 'dummy'), etype=etype)  # trigger edge computation

            # Extract node features for each edge manually
            src, dst = g.edges(etype=etype)
            h_src = g.nodes[src_type].data['h'][src]
            h_dst = g.nodes[dst_type].data['h'][dst]
            h_cat = torch.cat([h_src, h_dst], dim=1)

            score = self.mlp(h_cat).squeeze()

            if use_seed_score and 'seed_score' in g.edges[etype].data:
                score += g.edges[etype].data['seed_score'].squeeze()

            return score
