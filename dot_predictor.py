import dgl.function as fn
import torch.nn as nn


class DotPredictor(nn.Module):
    def forward(self, g, h, *args, **kwargs):
        """
        Forward method for link prediction, supporting both homogeneous
        and heterogeneous graph structures.

        Depending on the type of input `h`, this method calls the appropriate scoring function:
        - For a heterogeneous graph, it calls `forward_gnn` and expects `h` as a dictionary of
        node features and an `etype` in `kwargs`.
        - For a homogeneous graph, it calls `forward_subg` and expects `h` as a tensor, along
        with `u` and `v` in `kwargs`.

        Args:
            g (dgl.DGLGraph or dgl.DGLHeteroGraph): The input graph (homogeneous or heterogeneous).
            h (Union[torch.Tensor, dict]): Node embeddings. A tensor for homogeneous graphs,
            or a dictionary of node type to embeddings for heterogeneous graphs.
            *args: Additional positional arguments (not used directly).
            **kwargs: Keyword arguments. For heterogeneous graphs, expects `etype` (tuple).
            For homogeneous graphs, expects `u` and `v` (torch.Tensor).

        Returns:
            torch.Tensor: A 1D tensor of link prediction scores.
        """
        if isinstance(h, dict):
            return self.forward_gnn(g, h, kwargs['etype'])
        else:
            return self.forward_subg(g, h, kwargs['u'], kwargs['v'])

    def forward_gnn(self, g, h_dict, etype):
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
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return g.edges[etype].data['score'].squeeze()

    def forward_subg(self, g, h, u, v):
        """
        Compute link prediction scores for a homogeneous graph by calculating the dot product
        between source and target node embeddings.

        Args:
            g (dgl.DGLGraph): The homogeneous graph (unused here but kept
            for interface consistency).
            h (torch.Tensor): Node embeddings for the graph.
            u (torch.Tensor): Indices of source nodes for edges.
            v (torch.Tensor): Indices of target nodes for edges.

        Returns:
            torch.Tensor: A 1D tensor of scores for each (u, v) edge.
        """
        return (h[u] * h[v]).sum(dim=1)
