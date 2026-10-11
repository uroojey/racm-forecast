"""Unit tests for Day 5: Module 5 (CausalLearner) & Module 6 (CausalTemporalGT)."""

import pytest
import numpy as np
import networkx as nx
import torch

from causal.causal_learner import CausalLearner
from causal.racm_gt import (
    TemporalGraphAttention,
    GraphTransformerBlock,
    CausalTemporalGT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_causal_data():
    """Generates synthetic data from a known DAG: X0 -> X1 -> X2."""
    np.random.seed(42)
    n_samples = 300
    x0 = np.random.randn(n_samples)
    x1 = 0.8 * x0 + 0.2 * np.random.randn(n_samples)
    x2 = -0.6 * x1 + 0.3 * np.random.randn(n_samples)
    X = np.stack([x0, x1, x2], axis=1)
    return X


@pytest.fixture
def sample_kg():
    """Creates a basic NetworkX MultiDiGraph for testing graph merger."""
    kg = nx.MultiDiGraph()
    kg.add_node("AAPL", node_type="company")
    kg.add_node("MSFT", node_type="company")
    kg.add_node("GOOGL", node_type="company")
    return kg


# ── Module 5: CausalLearner Tests ─────────────────────────────────────────────

class TestCausalLearner:

    def test_fit_notears_shape(self, synthetic_causal_data):
        learner = CausalLearner()
        W = learner.fit(synthetic_causal_data, lambda1=0.01, max_iter=50, threshold=0.1)
        assert isinstance(W, np.ndarray)
        assert W.shape == (3, 3)
        # Self-loops should be zero
        np.testing.assert_array_equal(np.diag(W), np.zeros(3))

    def test_is_dag(self):
        learner = CausalLearner()
        # Strictly upper-triangular matrix is a valid DAG
        W_dag = np.array([
            [0.0, 0.5, 0.2],
            [0.0, 0.0, 0.8],
            [0.0, 0.0, 0.0]
        ])
        assert learner.is_dag(W_dag) is True

        # Matrix with cycle X0 -> X1 -> X0 is not a DAG
        W_cycle = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        assert learner.is_dag(W_cycle) is False

    def test_get_causal_edges(self):
        learner = CausalLearner()
        W = np.array([
            [0.0, 0.75, 0.05],
            [0.0, 0.0, -0.60],
            [0.0, 0.0, 0.0]
        ])
        node_names = ["AAPL", "MSFT", "GOOGL"]
        edges = learner.get_causal_edges(W, node_names=node_names, threshold=0.1)
        assert len(edges) == 2
        assert edges[0]["source"] == "AAPL"
        assert edges[0]["target"] == "MSFT"
        assert abs(edges[0]["weight"] - 0.75) < 1e-4

    def test_merge_with_knowledge_graph(self, sample_kg):
        learner = CausalLearner()
        W = np.array([
            [0.0, 0.75, 0.0],
            [0.0, 0.0, 0.65],
            [0.0, 0.0, 0.0]
        ])
        node_names = ["AAPL", "MSFT", "GOOGL"]
        enriched_kg = learner.merge_with_knowledge_graph(W, sample_kg, node_names=node_names, threshold=0.1)
        
        # Original kg should be untouched
        assert not sample_kg.has_edge("AAPL", "MSFT")
        
        # Enriched kg should have causal edges
        assert enriched_kg.has_edge("AAPL", "MSFT")
        assert enriched_kg.has_edge("MSFT", "GOOGL")
        
        edge_data = enriched_kg.get_edge_data("AAPL", "MSFT")
        first_edge = list(edge_data.values())[0]
        assert first_edge.get("edge_type") == "causes"
        assert abs(first_edge.get("weight") - 0.75) < 1e-4


# ── Module 6: RACM-GT Tests ───────────────────────────────────────────────────

class TestRACMGT:

    def test_temporal_graph_attention_shape(self):
        batch_size = 4
        num_nodes = 5
        d_model = 32
        n_heads = 4

        attn = TemporalGraphAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, num_nodes, d_model)
        causal_adj = torch.rand(num_nodes, num_nodes)

        out = attn(x, causal_adj)
        assert out.shape == (batch_size, num_nodes, d_model)

    def test_graph_transformer_block_shape(self):
        batch_size = 4
        num_nodes = 5
        d_model = 32

        block = GraphTransformerBlock(d_model=d_model, n_heads=4)
        x = torch.randn(batch_size, num_nodes, d_model)
        causal_adj = torch.rand(num_nodes, num_nodes)

        out = block(x, causal_adj)
        assert out.shape == (batch_size, num_nodes, d_model)

    def test_causal_temporal_gt_forward(self):
        batch_size = 8
        num_nodes = 6
        d_model = 64
        n_layers = 3

        model = CausalTemporalGT(
            num_nodes=num_nodes,
            d_model=d_model,
            n_heads=4,
            n_layers=n_layers,
            dropout=0.1,
        )

        x = torch.randn(batch_size, num_nodes, d_model)
        causal_adj = torch.rand(num_nodes, num_nodes)

        out = model(x, causal_adj)
        assert out.shape == (batch_size, num_nodes, d_model)

    def test_causal_temporal_gt_backward(self):
        batch_size = 2
        num_nodes = 4
        d_model = 16

        model = CausalTemporalGT(num_nodes=num_nodes, d_model=d_model, n_layers=2)
        x = torch.randn(batch_size, num_nodes, d_model, requires_grad=True)
        causal_adj = torch.rand(num_nodes, num_nodes)

        out = model(x, causal_adj)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_causal_temporal_gt_with_projector(self):
        batch_size = 2
        num_nodes = 4
        in_dim = 20
        d_model = 32

        model = CausalTemporalGT(
            num_nodes=num_nodes,
            d_model=d_model,
            in_dim=in_dim,
            n_layers=1,
        )

        x = torch.randn(batch_size, num_nodes, in_dim)
        causal_adj = torch.rand(num_nodes, num_nodes)

        out = model(x, causal_adj)
        assert out.shape == (batch_size, num_nodes, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
