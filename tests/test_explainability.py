"""Unit tests for Day 8: Module 11 (Explainability & Risk Analysis)."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from explainability.shap_explainer import SHAPExplainer
from explainability.graph_explainer import GraphExplainer
from explainability.counterfactual import CounterfactualGenerator


# ── Helper Models for Testing ─────────────────────────────────────────────────

class SimpleRegressor(nn.Module):
    """A trivial model for testing explainers."""
    def __init__(self, in_dim: int = 10, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleGNN(nn.Module):
    """A trivial GNN-like model for testing graph explainer."""
    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        return self.linear(x)


class DictModel(nn.Module):
    """A model returning a dict (like MultiTaskHead)."""
    def __init__(self, in_dim: int = 10):
        super().__init__()
        self.return_head = nn.Linear(in_dim, 1)
        self.direction_head = nn.Linear(in_dim, 3)

    def forward(self, x):
        return {
            "return": self.return_head(x),
            "direction": self.direction_head(x),
        }


# ── Module 11: SHAP Explainer Tests ──────────────────────────────────────────

class TestSHAPExplainer:

    def test_construction(self):
        model = SimpleRegressor(in_dim=10)
        feature_names = [f"feat_{i}" for i in range(10)]
        explainer = SHAPExplainer(model=model, feature_names=feature_names)
        assert explainer is not None

    def test_explain_top_k_with_simple_model(self):
        torch.manual_seed(42)
        model = SimpleRegressor(in_dim=5, out_dim=1)
        model.eval()
        feature_names = ["price", "volume", "rsi", "macd", "atr"]
        explainer = SHAPExplainer(model=model, feature_names=feature_names)

        X = np.random.randn(20, 5).astype(np.float32)

        try:
            top_features = explainer.explain_top_k(X, k=3, n_background=10)
            assert isinstance(top_features, list)
            assert len(top_features) <= 3
            for entry in top_features:
                assert "feature" in entry
                assert "importance" in entry
                assert "rank" in entry
        except ImportError:
            pytest.skip("shap not installed")

    def test_summary_report(self):
        model = SimpleRegressor(in_dim=5, out_dim=1)
        model.eval()
        explainer = SHAPExplainer(model=model, feature_names=["a", "b", "c", "d", "e"])
        X = np.random.randn(15, 5).astype(np.float32)

        try:
            report = explainer.summary_report(X, n_background=10)
            assert isinstance(report, str)
            assert len(report) > 0
        except ImportError:
            pytest.skip("shap not installed")


# ── Module 11: Graph Explainer Tests ──────────────────────────────────────────

class TestGraphExplainer:

    def test_construction(self):
        model = SimpleGNN(in_dim=8, out_dim=4)
        explainer = GraphExplainer(model=model, num_hops=2)
        assert explainer is not None

    def test_explain_node_gradient_fallback(self):
        model = SimpleGNN(in_dim=8, out_dim=4)
        model.eval()
        explainer = GraphExplainer(model=model, num_hops=2)

        num_nodes = 6
        x = torch.randn(num_nodes, 8)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)

        result = explainer.explain_node(x, edge_index, node_idx=0)
        assert "node_importance" in result
        assert "explained_node" in result
        assert result["explained_node"] == 0
        assert result["node_importance"].shape[0] == num_nodes

    def test_explain_graph(self):
        model = SimpleGNN(in_dim=4, out_dim=2)
        model.eval()
        explainer = GraphExplainer(model=model)

        num_nodes = 5
        x = torch.randn(num_nodes, 4)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        result = explainer.explain_graph(x, edge_index)
        assert "node_importance" in result
        assert "top_nodes" in result
        assert isinstance(result["top_nodes"], list)

    def test_summary_report(self):
        model = SimpleGNN(in_dim=4, out_dim=2)
        model.eval()
        explainer = GraphExplainer(model=model)

        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        node_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        report = explainer.summary_report(x, edge_index, node_names=node_names)
        assert isinstance(report, str)
        assert len(report) > 0


# ── Module 11: Counterfactual Generator Tests ────────────────────────────────

class TestCounterfactualGenerator:

    def test_generate_single_perturbation(self):
        model = SimpleRegressor(in_dim=5, out_dim=1)
        model.eval()
        feature_names = ["price", "volume", "rsi", "macd", "atr"]
        cfgen = CounterfactualGenerator(model=model, feature_names=feature_names)

        x = torch.randn(1, 5)
        result = cfgen.generate(x, feature_idx=2, delta=1.0)

        assert "original_prediction" in result
        assert "counterfactual_prediction" in result
        assert "delta" in result
        assert "perturbed_feature" in result

    def test_generate_with_replace_value(self):
        model = SimpleRegressor(in_dim=5, out_dim=1)
        model.eval()
        cfgen = CounterfactualGenerator(model=model, feature_names=["a", "b", "c", "d", "e"])

        x = torch.randn(1, 5)
        result = cfgen.generate(x, feature_idx=0, replace_value=0.0)

        assert "original_prediction" in result
        assert "counterfactual_prediction" in result

    def test_generate_batch(self):
        model = SimpleRegressor(in_dim=5, out_dim=1)
        model.eval()
        cfgen = CounterfactualGenerator(model=model)

        x = torch.randn(1, 5)
        results = cfgen.generate_batch(x, feature_indices=[0, 1, 2], deltas=[0.5, -0.5, 1.0])

        assert isinstance(results, list)
        assert len(results) == 3

    def test_what_if_no_event(self):
        model = DictModel(in_dim=10)
        model.eval()
        cfgen = CounterfactualGenerator(model=model)

        x = torch.randn(1, 10)
        event_features = [7, 8, 9]  # Last 3 features are "event" features
        result = cfgen.what_if_no_event(x, event_feature_indices=event_features)

        assert "original_prediction" in result
        assert "counterfactual_prediction" in result
        assert "delta" in result

    def test_summary_report(self):
        model = SimpleRegressor(in_dim=5, out_dim=1)
        model.eval()
        cfgen = CounterfactualGenerator(model=model, feature_names=["a", "b", "c", "d", "e"])

        x = torch.randn(1, 5)
        report = cfgen.summary_report(x, feature_indices=[0, 1, 2])

        assert isinstance(report, str)
        assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
