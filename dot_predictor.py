import dgl.function as fn
import torch.nn as nn


class DotPredictor(nn.Module):
    def forward(self, g, h_dict, etype):
        with g.local_scope():
            src_type, _, dst_type = g.to_canonical_etype(etype)
            g.nodes[src_type].data['h'] = h_dict[src_type]
            g.nodes[dst_type].data['h'] = h_dict[dst_type]
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return g.edges[etype].data['score'].squeeze()
