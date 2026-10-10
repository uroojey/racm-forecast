"""Unit tests for Module 4 — Dynamic Financial Knowledge Graph."""

import pytest
import networkx as nx
import torch
import os
import tempfile
from torch_geometric.data import Data

from knowledge_graph.schema import NodeType, EdgeType
from knowledge_graph.builder import DynamicGraphBuilder
from knowledge_graph.visualizer import GraphVisualizer
from knowledge_graph.embedder import GraphEmbedder

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_events():
    return [
        {
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "event_type": "earnings",
            "description": "Earnings beat",
            "impact": "positive",
            "impact_magnitude": 0.8,
            "date": "2024-01-01",
            "source_text": "Apple beat earnings...",
            "confidence": 0.95
        },
        {
            "company": "Microsoft",
            "ticker": "MSFT",
            "event_type": "product_launch",
            "description": "New AI product",
            "impact": "positive",
            "impact_magnitude": 0.9,
            "date": "2024-01-01",
            "source_text": "Microsoft launched...",
            "confidence": 0.99
        }
    ]

@pytest.fixture
def base_builder():
    builder = DynamicGraphBuilder()
    builder.add_company(ticker="AAPL", name="Apple Inc.", sector="Technology")
    builder.add_company(ticker="MSFT", name="Microsoft", sector="Technology")
    builder.add_company(ticker="GOOGL", name="Alphabet", sector="Technology")
    builder.add_competitor_edge("AAPL", "MSFT")
    return builder


# ── Tests for Schema ──────────────────────────────────────────────────────────

def test_schema_enums():
    assert NodeType.COMPANY == "company"
    assert EdgeType.BELONGS_TO == "belongs_to"


# ── Tests for Builder ─────────────────────────────────────────────────────────

class TestDynamicGraphBuilder:
    def test_add_company_and_sector(self, base_builder):
        graph = base_builder.base_graph
        
        # Check nodes
        assert "AAPL" in graph.nodes
        assert graph.nodes["AAPL"]["node_type"] == NodeType.COMPANY
        assert "Technology" in graph.nodes
        assert graph.nodes["Technology"]["node_type"] == NodeType.SECTOR
        
        # Check edges
        assert graph.has_edge("AAPL", "Technology")
        edge_data = graph.get_edge_data("AAPL", "Technology")
        # MultiDiGraph returns a dict of edges between two nodes (keyed by edge key, usually 0, 1, ...)
        assert list(edge_data.values())[0]["edge_type"] == EdgeType.BELONGS_TO

    def test_add_competitor_edge(self, base_builder):
        graph = base_builder.base_graph
        assert graph.has_edge("AAPL", "MSFT")
        assert list(graph.get_edge_data("AAPL", "MSFT").values())[0]["edge_type"] == EdgeType.COMPETES_WITH

    def test_build_snapshot(self, base_builder, sample_events):
        date = "2024-01-01"
        snapshot = base_builder.build_snapshot(date=date, events=sample_events)
        
        assert snapshot is not None
        assert base_builder.get_snapshot(date) == snapshot
        
        # Check if event nodes were added
        event_nodes = [n for n, d in snapshot.nodes(data=True) if d.get("node_type") == NodeType.EVENT]
        assert len(event_nodes) == 2
        
        # Check if HAS_EVENT edges exist (from Company to Event)
        # We need to find the specific event node ID, which is likely generated in the builder.
        # Let's check if AAPL has any outgoing edges of type HAS_EVENT
        has_event_edges = [
            (u, v) for u, v, d in snapshot.edges(data=True) 
            if d.get("edge_type") == EdgeType.HAS_EVENT and u == "AAPL"
        ]
        assert len(has_event_edges) >= 1

    def test_build_snapshot_missing_ticker(self, base_builder):
        # Event with a ticker not in base graph
        events = [{
            "company": "Tesla",
            "ticker": "TSLA",
            "event_type": "product_launch",
            "description": "New car",
            "impact": "positive",
            "impact_magnitude": 0.8,
            "date": "2024-01-01",
            "source_text": "...",
            "confidence": 0.9
        }]
        # It should handle it gracefully (e.g., by adding the company node on the fly, or skipping)
        snapshot = base_builder.build_snapshot(date="2024-01-01", events=events)
        assert "TSLA" in snapshot.nodes


# ── Tests for Visualizer ──────────────────────────────────────────────────────

class TestGraphVisualizer:
    def test_visualize_generates_html(self, base_builder):
        visualizer = GraphVisualizer()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            visualizer.visualize(base_builder.base_graph, output_path=tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path, "r") as f:
                content = f.read()
                assert "<html" in content.lower()
                assert "network" in content.lower()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ── Tests for Embedder ────────────────────────────────────────────────────────

class TestGraphEmbedder:
    def test_nx_to_pyg(self, base_builder):
        feature_dim = 16
        data = GraphEmbedder.nx_to_pyg(base_builder.base_graph, feature_dim=feature_dim)
        
        assert isinstance(data, Data)
        assert data.x is not None
        assert data.edge_index is not None
        
        num_nodes = len(base_builder.base_graph.nodes)
        assert data.x.shape == (num_nodes, feature_dim)
        
        # Check edge_index shape: (2, num_edges)
        num_edges = len(base_builder.base_graph.edges)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == num_edges

    def test_forward_pass(self):
        in_channels = 16
        hidden_channels = 32
        out_channels = 64
        num_nodes = 10
        num_edges = 15
        
        embedder = GraphEmbedder(in_channels, hidden_channels, out_channels)
        
        # Create dummy PyG Data
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        out = embedder(x, edge_index)
        
        assert out.shape == (num_nodes, out_channels)
        
    def test_backward_pass(self):
        in_channels = 16
        embedder = GraphEmbedder(in_channels, 32, 64)
        x = torch.randn(10, in_channels, requires_grad=True)
        edge_index = torch.randint(0, 10, (2, 15))
        
        out = embedder(x, edge_index)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
