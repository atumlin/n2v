"""
GraphNeuralNetwork: a high-level wrapper for GNN reachability.

Owns an ordered stack of graph-conv layer specs together with the graph
structure (adjacency for GCN/SAGE, or edge list + edge features for GINE) and
exposes ``evaluate`` (concrete forward) and ``reach`` (set propagation),
mirroring the role of :class:`n2v.nn.NeuralNetwork` for feed-forward models.

Layer specs are the dataclasses produced by :func:`n2v.utils.load_gnn_mat`
(:class:`GCNLayerSpec`, :class:`SAGELayerSpec`, :class:`GINELayerSpec`); the
wrapper dispatches per spec type to the corresponding reach op.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from n2v.sets import GraphStar
from n2v.utils.gnn_loader import (
    GNNModel, GCNLayerSpec, SAGELayerSpec, GINELayerSpec, load_gnn_mat,
)
from n2v.utils.subgraph import khop_subgraph_matrix, khop_subgraph_edges
from n2v.utils.gnn_falsify import falsify_node_bounds, FalsifyResult
from n2v.nn.layer_ops.gcn_reach import (
    gcn_graph_star, gcn_evaluate, relu_graph_star,
)
from n2v.nn.layer_ops.sage_reach import sage_graph_star, sage_evaluate
from n2v.nn.layer_ops.gine_reach import gine_graph_star, gine_evaluate


@dataclass
class VerifyResult:
    """Verdict from :meth:`GraphNeuralNetwork.verify`."""
    status: str                              # 'verified' | 'falsified' | 'unknown'
    reach_set: "GraphStar" = None
    counterexample: "FalsifyResult" = None   # populated when status == 'falsified'


@dataclass
class SubgraphResult:
    """Per-target output of :meth:`GraphNeuralNetwork.reach_subgraph`."""
    target_node: int          # original 0-indexed node
    target_local_idx: int     # its row index within ``output``
    output: GraphStar         # reach set over the subgraph nodes
    n_sub_nodes: int
    n_sub_edges: int

    def target_ranges(self, **kwargs):
        """(lb, ub) feature vectors for the target node only."""
        lb, ub = self.output.get_ranges(**kwargs)
        return lb[self.target_local_idx], ub[self.target_local_idx]


class GraphNeuralNetwork:
    """Reachability wrapper around a stack of graph-convolution layers.

    Args:
        layers: Ordered layer specs (all the same family: GCN, SAGE, or GINE).
        adjacency: Dense adjacency for GCN (normalized) or SAGE (binary).
        has_relu: Apply ReLU between GCN/SAGE layers (ignored for GINE, which
            carries its own internal ReLUs).
        edge_index: (2, m) 0-indexed edge list (GINE only).
        E: Edge features (m, E_in) (GINE only).
        edge_weights: Per-edge aggregation weights (m,) (GINE only).
        gine_variant: 'hugine' (gine_pretrain) or 'pyg' (gine_conv).
    """

    def __init__(
        self,
        layers: Sequence,
        adjacency: Optional[np.ndarray] = None,
        has_relu: bool = True,
        edge_index: Optional[np.ndarray] = None,
        E: Optional[np.ndarray] = None,
        edge_weights: Optional[np.ndarray] = None,
        gine_variant: str = "hugine",
    ):
        if len(layers) == 0:
            raise ValueError("GraphNeuralNetwork requires at least one layer")
        kinds = {type(l).__name__ for l in layers}
        if len(kinds) != 1:
            raise ValueError(f"All layers must be the same type; got {sorted(kinds)}")

        self.layers = list(layers)
        self.layer_kind = next(iter(kinds))
        self.adjacency = None if adjacency is None else np.asarray(adjacency, dtype=np.float64)
        self.has_relu = has_relu
        self.edge_index = None if edge_index is None else np.asarray(edge_index)
        self.E = None if E is None else np.asarray(E, dtype=np.float64)
        self.edge_weights = None if edge_weights is None else np.asarray(edge_weights, dtype=np.float64)
        self.gine_variant = gine_variant

        if self.layer_kind == "GINELayerSpec":
            if self.edge_index is None or self.E is None:
                raise ValueError("GINE network requires edge_index and E")
        else:
            if self.adjacency is None:
                raise ValueError(f"{self.layer_kind} network requires an adjacency matrix")

    # ----------------------------------------------------------------- builders

    @classmethod
    def from_model(cls, model: GNNModel) -> "GraphNeuralNetwork":
        """Build from a parsed :class:`GNNModel` (see :func:`load_gnn_mat`)."""
        return cls(
            layers=model.layers,
            adjacency=model.adjacency,
            has_relu=model.has_relu,
            edge_index=model.edge_index,
            E=model.E,
            edge_weights=model.edge_weights,
            gine_variant=model.gine_variant or "hugine",
        )

    @classmethod
    def from_mat(cls, path) -> "GraphNeuralNetwork":
        """Load a .mat checkpoint and wrap it."""
        return cls.from_model(load_gnn_mat(path))

    # --------------------------------------------------------------- properties

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        return (
            f"GraphNeuralNetwork(kind={self.layer_kind}, layers={self.num_layers}, "
            f"variant={self.gine_variant if self.layer_kind == 'GINELayerSpec' else 'n/a'})"
        )

    # ------------------------------------------------------------ forward eval

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Concrete forward pass over node features ``X`` (N, F_in)."""
        Y = np.asarray(X, dtype=np.float64)
        n = self.num_layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GCNLayerSpec):
                Y = gcn_evaluate(Y, self.adjacency, layer.W, layer.b)
                if self.has_relu:
                    Y = np.maximum(Y, 0.0)
            elif isinstance(layer, SAGELayerSpec):
                Y = sage_evaluate(Y, self.adjacency, layer.W_node, layer.W_edge, layer.b)
                if self.has_relu:
                    Y = np.maximum(Y, 0.0)
            elif isinstance(layer, GINELayerSpec):
                out_relu = (self.gine_variant == "hugine") and (i < n - 1)
                Y = gine_evaluate(Y, self.E, self.edge_index, layer,
                                  self.edge_weights, self.gine_variant, out_relu)
            else:
                raise TypeError(f"Unsupported layer spec: {type(layer).__name__}")
        return Y

    # --------------------------------------------------------------- reach

    def reach(
        self,
        input_set: GraphStar,
        method: str = "approx",
        relax_factor: float = 0.5,
        lp_solver: str = "default",
    ) -> List[GraphStar]:
        """Propagate a GraphStar through the layer stack.

        Args:
            input_set: Input GraphStar over node features.
            method: 'approx' (triangle-relaxation ReLU; single set) — the only
                mode wired for GNNs today.
            relax_factor / lp_solver: forwarded to the ReLU relaxation.

        Returns:
            List of output GraphStars (length 1 under approx-star).
        """
        if not isinstance(input_set, GraphStar):
            raise TypeError(f"reach expects a GraphStar, got {type(input_set).__name__}")
        if method != "approx":
            raise ValueError(f"GNN reach supports method='approx' only; got '{method}'")

        n = self.num_layers
        sets: List[GraphStar] = [input_set]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GCNLayerSpec):
                sets = [gcn_graph_star(s, layer.W, layer.b, self.adjacency) for s in sets]
                if self.has_relu:
                    sets = self._relu(sets, relax_factor, lp_solver)
            elif isinstance(layer, SAGELayerSpec):
                sets = [sage_graph_star(s, layer.W_node, layer.W_edge, layer.b, self.adjacency)
                        for s in sets]
                if self.has_relu:
                    sets = self._relu(sets, relax_factor, lp_solver)
            elif isinstance(layer, GINELayerSpec):
                out_relu = (self.gine_variant == "hugine") and (i < n - 1)
                sets = [
                    gine_graph_star(s, self.E, self.edge_index, layer, self.edge_weights,
                                    self.gine_variant, apply_output_relu=out_relu,
                                    relax_factor=relax_factor, lp_solver=lp_solver)
                    for s in sets
                ]
            else:
                raise TypeError(f"Unsupported layer spec: {type(layer).__name__}")
        return sets

    @staticmethod
    def _relu(sets, relax_factor, lp_solver):
        out = []
        for s in sets:
            out.extend(relu_graph_star(s, method="approx",
                                       relax_factor=relax_factor, lp_solver=lp_solver))
        return out

    # ------------------------------------------------------- verify + falsify

    def verify(
        self,
        input_set: GraphStar,
        spec_lb: np.ndarray,
        spec_ub: np.ndarray,
        target_nodes: Optional[Sequence[int]] = None,
        falsify: bool = True,
        relax_factor: float = 0.5,
        lp_solver: str = "default",
        **falsify_kwargs,
    ) -> "VerifyResult":
        """Verify outputs stay in [spec_lb, spec_ub]; falsify the 'unknown' gap.

        Runs sound reachability first.  If the reach set proves the box holds,
        returns 'verified'.  Otherwise (when ``falsify``) searches the input box
        for a concrete counterexample; a hit returns 'falsified' with a witness,
        a miss returns 'unknown'.

        ``input_set`` must be a box GraphStar (built via ``from_bounds``); its
        own range supplies the input box handed to the falsifier.
        """
        from n2v.utils.gnn_verify import verify_node_bounds

        out = self.reach(input_set, relax_factor=relax_factor, lp_solver=lp_solver)[0]
        status = verify_node_bounds([out], spec_lb, spec_ub, target_nodes)
        if status == "verified" or not falsify:
            return VerifyResult(status=status, reach_set=out)

        in_lb, in_ub = input_set.get_ranges()
        res = falsify_node_bounds(self.evaluate, in_lb, in_ub, spec_lb, spec_ub,
                                  target_nodes=target_nodes, **falsify_kwargs)
        if res.found:
            return VerifyResult(status="falsified", reach_set=out, counterexample=res)
        return VerifyResult(status="unknown", reach_set=out)

    # --------------------------------------------------------- subgraph reach

    def reach_subgraph(
        self,
        input_set: GraphStar,
        target_nodes: Sequence[int],
        method: str = "approx",
        relax_factor: float = 0.5,
        lp_solver: str = "default",
    ) -> List[SubgraphResult]:
        """Per-target k-hop subgraph reachability (scalable, exact per target).

        For each target node, extracts its k-hop neighborhood (k = number of
        message-passing layers), prunes the input GraphStar to that subset, and
        runs reach on the small sub-network.  The target node's output is
        mathematically identical to full-graph reach because message passing is
        k-hop local — only boundary nodes compute unused intermediate values.

        Args:
            input_set: Full-graph input GraphStar (N nodes).
            target_nodes: 0-indexed node indices to verify.
            method / relax_factor / lp_solver: forwarded to :meth:`reach`.

        Returns:
            One :class:`SubgraphResult` per target node.
        """
        if not isinstance(input_set, GraphStar):
            raise TypeError(f"reach_subgraph expects a GraphStar, got {type(input_set).__name__}")

        k = self.num_layers                      # all layers are message-passing
        is_gine = self.layer_kind == "GINELayerSpec"
        results: List[SubgraphResult] = []

        for t in target_nodes:
            if is_gine:
                sub_nodes, sub_ei, sub_E, sub_ew, t_local = khop_subgraph_edges(
                    int(t), k, self.edge_index, self.E, self.edge_weights)
                sub_gs = input_set.extract_subgraph(sub_nodes)
                sub_net = GraphNeuralNetwork(
                    self.layers, edge_index=sub_ei, E=sub_E, edge_weights=sub_ew,
                    gine_variant=self.gine_variant, has_relu=self.has_relu)
                n_sub_edges = sub_ei.shape[1]
            else:
                sub_nodes, sub_A, t_local = khop_subgraph_matrix(int(t), k, self.adjacency)
                sub_gs = input_set.extract_subgraph(sub_nodes, sub_adjacency=sub_A)
                sub_net = GraphNeuralNetwork(self.layers, adjacency=sub_A, has_relu=self.has_relu)
                n_sub_edges = int(np.count_nonzero(sub_A)) - len(sub_nodes)

            out = sub_net.reach(sub_gs, method=method,
                                relax_factor=relax_factor, lp_solver=lp_solver)[0]
            results.append(SubgraphResult(
                target_node=int(t), target_local_idx=t_local, output=out,
                n_sub_nodes=len(sub_nodes), n_sub_edges=n_sub_edges))
        return results
