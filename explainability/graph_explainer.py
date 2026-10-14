"""Module 11: Graph-level Interpretability for RACM-Forecast.

Implements node- and edge-level graph explanations using GNNExplainer (when available)
with a gradient-based sensitivity fallback.
"""

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    _PYG_EXPLAIN_AVAILABLE = True
except ImportError:
    Explainer = None
    GNNExplainer = None
    _PYG_EXPLAIN_AVAILABLE = False


class GraphExplainer:
    """Graph neural network explainer for node- and edge-level importance.

    Identifies key market entities (nodes) and relational spillover channels (edges)
    influencing asset return and volatility forecasts. Supports PyTorch Geometric's
    Explainer / GNNExplainer when available, with a robust gradient-based sensitivity
    fallback for arbitrary PyTorch graph architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        num_hops: int = 2,
    ) -> None:
        """Initialize the GraphExplainer.

        Args:
            model: PyTorch graph neural network model to explain.
            num_hops: Number of hops in the computational graph neighborhood.
        """
        self.model = model
        self.num_hops = int(num_hops)

    def _get_k_hop_mask(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        node_idx: int,
        num_hops: int,
    ) -> torch.Tensor:
        """Compute a boolean mask of nodes within k hops of node_idx."""
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        mask[node_idx] = True
        if num_hops <= 0 or edge_index.numel() == 0:
            return mask

        src = edge_index[0]
        dst = edge_index[1]
        current_nodes = {node_idx}

        for _ in range(num_hops):
            next_nodes = set()
            for u in current_nodes:
                nbrs_out = dst[src == u].tolist()
                nbrs_in = src[dst == u].tolist()
                next_nodes.update(nbrs_out)
                next_nodes.update(nbrs_in)
            current_nodes = next_nodes
            for n in current_nodes:
                if 0 <= n < num_nodes:
                    mask[n] = True

        return mask

    def _get_device(self) -> torch.device:
        """Get model device."""
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def _explain_node_gradient(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
    ) -> Dict[str, Any]:
        """Gradient-based sensitivity explanation fallback for a specific node.

        Computes the gradient of the target node output w.r.t. input node features
        and edge connections.

        Args:
            x: Node feature tensor of shape (num_nodes, num_features).
            edge_index: Graph connectivity tensor of shape (2, num_edges).
            node_idx: Index of node to explain.

        Returns:
            Dictionary with node_importance, edge_importance, and explained_node.
        """
        device = self._get_device()
        num_nodes = x.size(0)
        num_edges = edge_index.size(1) if (edge_index.ndim == 2 and edge_index.size(0) == 2) else 0

        if not (0 <= node_idx < num_nodes):
            raise ValueError(f"node_idx {node_idx} out of range [0, {num_nodes - 1}]")

        x_input = x.to(device).clone().detach().requires_grad_(True)
        edge_index_in = edge_index.to(device)
        edge_weight = torch.ones(num_edges, dtype=x.dtype, device=device, requires_grad=True)

        was_training = self.model.training
        self.model.eval()

        try:
            # Test model signature for edge_weight support
            out = None
            edge_weight_supported = False
            try:
                out = self.model(x_input, edge_index_in, edge_weight)
                edge_weight_supported = True
            except (TypeError, ValueError):
                try:
                    out = self.model(x_input, edge_index_in)
                except TypeError:
                    out = self.model(x_input)

            # Extract output prediction for target node
            if isinstance(out, dict):
                if "return" in out:
                    out_tensor = out["return"]
                else:
                    out_tensor = next(iter(out.values()))
            else:
                out_tensor = out

            if out_tensor.ndim >= 2 and out_tensor.size(0) == num_nodes:
                target = out_tensor[node_idx]
            elif out_tensor.ndim == 1 and out_tensor.size(0) == num_nodes:
                target = out_tensor[node_idx]
            else:
                target = out_tensor

            # Zero model gradients before differentiating prediction
            self.model.zero_grad(set_to_none=True)
            loss = target.sum()
            loss.backward()
            self.model.zero_grad(set_to_none=True)

            # Node importance from gradient magnitude
            if x_input.grad is not None:
                node_importance = torch.norm(x_input.grad, p=2, dim=-1)
            else:
                node_importance = torch.zeros(num_nodes, device=device)

            # Edge importance
            if edge_weight_supported and edge_weight.grad is not None:
                edge_importance = edge_weight.grad.abs()
            elif num_edges > 0:
                # Approximate edge importance via endpoint node importances
                src = edge_index_in[0]
                dst = edge_index_in[1]
                edge_importance = 0.5 * (node_importance[src] + node_importance[dst])
            else:
                edge_importance = torch.zeros(0, device=device)

        finally:
            if was_training:
                self.model.train()

        return {
            "node_importance": node_importance.detach(),
            "edge_importance": edge_importance.detach(),
            "explained_node": int(node_idx),
        }

    def explain_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
    ) -> Dict[str, Any]:
        """Explain the prediction for a specific node in the graph.

        Uses PyTorch Geometric's Explainer with GNNExplainer if available and compatible.
        Otherwise falls back to the gradient-based sensitivity approach.

        Args:
            x: Node feature tensor of shape (num_nodes, num_features).
            edge_index: Edge index tensor of shape (2, num_edges).
            node_idx: Target node index to explain.

        Returns:
            Dictionary containing:
                - 'node_importance': torch.Tensor of shape (num_nodes,)
                - 'edge_importance': torch.Tensor of shape (num_edges,)
                - 'explained_node': int index of explained node
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1) if (edge_index.ndim == 2 and edge_index.size(0) == 2) else 0

        if _PYG_EXPLAIN_AVAILABLE and Explainer is not None and GNNExplainer is not None:
            try:
                explainer = Explainer(
                    model=self.model,
                    algorithm=GNNExplainer(epochs=100),
                    explanation_type="model",
                    node_mask_type="attributes",
                    edge_mask_type="object",
                    model_config=dict(
                        mode="regression",
                        task_level="node",
                        return_type="raw",
                    ),
                )
                explanation = explainer(x, edge_index, index=node_idx)

                # Extract node importance
                if hasattr(explanation, "node_mask") and explanation.node_mask is not None:
                    node_mask = explanation.node_mask
                    if node_mask.ndim > 1:
                        node_imp = node_mask.norm(p=2, dim=-1)
                    else:
                        node_imp = node_mask
                else:
                    node_imp = torch.zeros(num_nodes, device=x.device)

                # Extract edge importance
                if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None:
                    edge_imp = explanation.edge_mask
                else:
                    edge_imp = torch.zeros(num_edges, device=x.device)

                return {
                    "node_importance": node_imp.detach(),
                    "edge_importance": edge_imp.detach(),
                    "explained_node": int(node_idx),
                }
            except Exception:
                # Graceful fallback to gradient-based approach
                pass

        return self._explain_node_gradient(x, edge_index, node_idx)

    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Aggregate node-level importance across all nodes in the graph.

        Args:
            x: Node feature tensor of shape (num_nodes, num_features).
            edge_index: Edge index tensor of shape (2, num_edges).
            top_k: Number of top nodes and edges to extract.

        Returns:
            Dictionary containing:
                - 'node_importance': torch.Tensor aggregated per-node importance
                - 'edge_importance': torch.Tensor aggregated per-edge importance
                - 'top_nodes': List[int] indices of most important nodes
                - 'top_edges': List[Tuple[int, int]] most important (src, dst) edges
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1) if (edge_index.ndim == 2 and edge_index.size(0) == 2) else 0

        agg_node_importance = torch.zeros(num_nodes, device=x.device)
        agg_edge_importance = torch.zeros(num_edges, device=x.device)

        if num_nodes == 0:
            return {
                "node_importance": agg_node_importance,
                "edge_importance": agg_edge_importance,
                "top_nodes": [],
                "top_edges": [],
            }

        # Aggregate explanation across each node
        for i in range(num_nodes):
            res = self.explain_node(x, edge_index, node_idx=i)
            agg_node_importance += res["node_importance"]
            if num_edges > 0 and "edge_importance" in res:
                agg_edge_importance += res["edge_importance"]

        agg_node_importance = agg_node_importance / float(num_nodes)
        if num_edges > 0:
            agg_edge_importance = agg_edge_importance / float(num_nodes)

        # Rank nodes
        sorted_node_indices = torch.argsort(agg_node_importance, descending=True).tolist()
        k_nodes = min(max(1, top_k), num_nodes)
        top_nodes = sorted_node_indices[:k_nodes]

        # Rank edges
        top_edges: List[Tuple[int, int]] = []
        if num_edges > 0:
            sorted_edge_indices = torch.argsort(agg_edge_importance, descending=True).tolist()
            k_edges = min(max(1, top_k), num_edges)
            for e_idx in sorted_edge_indices[:k_edges]:
                src = int(edge_index[0, e_idx].item())
                dst = int(edge_index[1, e_idx].item())
                top_edges.append((src, dst))

        return {
            "node_importance": agg_node_importance,
            "edge_importance": agg_edge_importance,
            "top_nodes": top_nodes,
            "top_edges": top_edges,
        }

    def summary_report(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_names: Optional[List[str]] = None,
    ) -> str:
        """Generate a formatted string report of the top-5 most important nodes and edges.

        Args:
            x: Node feature tensor of shape (num_nodes, num_features).
            edge_index: Edge index tensor of shape (2, num_edges).
            node_names: Optional list of human-readable node names (e.g., ticker symbols).

        Returns:
            Formatted string report.
        """
        graph_exp = self.explain_graph(x, edge_index, top_k=5)
        top_nodes = graph_exp["top_nodes"]
        top_edges = graph_exp["top_edges"]
        node_imp = graph_exp["node_importance"]
        edge_imp = graph_exp["edge_importance"]

        lines = [
            "=" * 68,
            "              Graph Explanation Summary Report",
            "=" * 68,
            f"Total Nodes: {x.size(0)} | Total Edges: {edge_index.size(1) if edge_index.ndim == 2 else 0}",
            "-" * 68,
            "Top Important Nodes:",
            f"{'Rank':<6} | {'Node':<34} | {'Importance Score':<18}",
            "-" * 68,
        ]

        for rank, node_id in enumerate(top_nodes, start=1):
            if node_names and 0 <= node_id < len(node_names):
                label = f"{node_names[node_id]} (node {node_id})"
            else:
                label = f"Node {node_id}"
            score = float(node_imp[node_id].item())
            lines.append(f"{rank:<6} | {label:<34} | {score:<18.6f}")

        lines.extend([
            "-" * 68,
            "Top Important Edges:",
            f"{'Rank':<6} | {'Edge (Source -> Destination)':<34} | {'Importance Score':<18}",
            "-" * 68,
        ])

        if top_edges:
            for rank, (src, dst) in enumerate(top_edges, start=1):
                if node_names and 0 <= src < len(node_names) and 0 <= dst < len(node_names):
                    edge_label = f"{node_names[src]} -> {node_names[dst]}"
                else:
                    edge_label = f"Node {src} -> Node {dst}"

                # Find edge score if available
                score_str = "N/A"
                if edge_index.ndim == 2 and edge_index.size(1) > 0:
                    src_matches = (edge_index[0] == src) & (edge_index[1] == dst)
                    if src_matches.any():
                        e_idx = src_matches.nonzero(as_tuple=True)[0][0].item()
                        score_str = f"{float(edge_imp[e_idx].item()):.6f}"

                lines.append(f"{rank:<6} | {edge_label:<34} | {score_str:<18}")
        else:
            lines.append("  No edges present or recorded in graph.")

        lines.append("=" * 68)
        return "\n".join(lines)
