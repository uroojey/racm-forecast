"""Unit tests for Day 6: Module 7 (MarketRegimeDetector) & Module 8 (MoEFusion)."""

import pytest
import numpy as np
import torch

from regime.regime_detector import MarketRegime, MarketRegimeDetector
from regime.moe_fusion import ExpertNetwork, RegimeGatingNetwork, MoEFusion


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_regime_data():
    """Generates synthetic time-series with 4 distinct regime distributions:
    - Bull: high positive mean, low variance
    - Bear: negative mean, moderate variance
    - High Volatility: zero mean, high variance
    - Crisis: strong negative mean, extreme variance
    """
    np.random.seed(42)
    n_per_regime = 100
    
    # Feature 0: returns, Feature 1: volatility estimate
    bull = np.column_stack([
        np.random.normal(loc=0.05, scale=0.01, size=n_per_regime),
        np.random.normal(loc=0.01, scale=0.002, size=n_per_regime)
    ])
    bear = np.column_stack([
        np.random.normal(loc=-0.03, scale=0.02, size=n_per_regime),
        np.random.normal(loc=0.03, scale=0.005, size=n_per_regime)
    ])
    high_vol = np.column_stack([
        np.random.normal(loc=0.00, scale=0.04, size=n_per_regime),
        np.random.normal(loc=0.06, scale=0.01, size=n_per_regime)
    ])
    crisis = np.column_stack([
        np.random.normal(loc=-0.08, scale=0.06, size=n_per_regime),
        np.random.normal(loc=0.12, scale=0.02, size=n_per_regime)
    ])

    X = np.vstack([bull, bear, high_vol, crisis])
    return X


# ── Module 7: MarketRegimeDetector Tests ──────────────────────────────────────

class TestMarketRegimeDetector:

    def test_market_regime_enum(self):
        assert MarketRegime.BULL == 0
        assert MarketRegime.BEAR == 1
        assert MarketRegime.HIGH_VOLATILITY == 2
        assert MarketRegime.CRISIS == 3

    def test_fit_and_predict(self, synthetic_regime_data):
        detector = MarketRegimeDetector(n_regimes=4, random_state=42)
        detector.fit(synthetic_regime_data)
        
        preds = detector.predict(synthetic_regime_data)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(synthetic_regime_data),)
        
        # Check all predicted values are within {0, 1, 2, 3}
        unique_states = np.unique(preds)
        assert set(unique_states).issubset({0, 1, 2, 3})

    def test_predict_proba(self, synthetic_regime_data):
        detector = MarketRegimeDetector(n_regimes=4, random_state=42)
        detector.fit(synthetic_regime_data)

        proba = detector.predict_proba(synthetic_regime_data)
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (len(synthetic_regime_data), 4)

        # Probabilities should sum to 1 across regimes
        row_sums = np.sum(proba, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_detect_change_points(self, synthetic_regime_data):
        detector = MarketRegimeDetector(n_regimes=4, random_state=42)
        detector.fit(synthetic_regime_data)

        bkps = detector.detect_change_points(synthetic_regime_data, n_bkps=3)
        assert isinstance(bkps, list)
        assert len(bkps) > 0
        # Breakpoints should be within the time series bounds
        for b in bkps:
            assert 0 <= b <= len(synthetic_regime_data)


# ── Module 8: MoEFusion Tests ─────────────────────────────────────────────────

class TestMoEFusion:

    def test_expert_network_shape(self):
        d_model = 32
        expert = ExpertNetwork(d_model=d_model, hidden_dim=64, dropout=0.1)
        
        # Test 2D input (batch_size, d_model)
        x_2d = torch.randn(8, d_model)
        out_2d = expert(x_2d)
        assert out_2d.shape == (8, d_model)

        # Test 3D input (batch_size, num_nodes, d_model)
        x_3d = torch.randn(4, 10, d_model)
        out_3d = expert(x_3d)
        assert out_3d.shape == (4, 10, d_model)

    def test_regime_gating_network(self):
        d_model = 32
        num_experts = 4
        num_regimes = 4

        gating = RegimeGatingNetwork(d_model=d_model, num_experts=num_experts, num_regimes=num_regimes)

        # Without regime_probs
        x = torch.randn(8, d_model)
        gates = gating(x)
        assert gates.shape == (8, num_experts)
        np.testing.assert_allclose(gates.sum(dim=-1).detach().numpy(), 1.0, atol=1e-4)

        # With regime_probs
        regime_probs = torch.softmax(torch.randn(8, num_regimes), dim=-1)
        gates_conditioned = gating(x, regime_probs=regime_probs)
        assert gates_conditioned.shape == (8, num_experts)
        np.testing.assert_allclose(gates_conditioned.sum(dim=-1).detach().numpy(), 1.0, atol=1e-4)

    def test_moe_fusion_forward_2d(self):
        batch_size = 16
        d_model = 64
        num_experts = 4

        moe = MoEFusion(d_model=d_model, num_experts=num_experts)
        x = torch.randn(batch_size, d_model)
        regime_probs = torch.softmax(torch.randn(batch_size, num_experts), dim=-1)

        out, gates = moe(x, regime_probs=regime_probs)
        assert out.shape == (batch_size, d_model)
        assert gates.shape == (batch_size, num_experts)

    def test_moe_fusion_forward_3d(self):
        batch_size = 4
        num_nodes = 8
        d_model = 32
        num_experts = 4

        moe = MoEFusion(d_model=d_model, num_experts=num_experts)
        x = torch.randn(batch_size, num_nodes, d_model)
        regime_probs = torch.softmax(torch.randn(batch_size, num_nodes, num_experts), dim=-1)

        out, gates = moe(x, regime_probs=regime_probs)
        assert out.shape == (batch_size, num_nodes, d_model)
        assert gates.shape == (batch_size, num_nodes, num_experts)

    def test_moe_fusion_backward(self):
        batch_size = 4
        d_model = 32
        moe = MoEFusion(d_model=d_model, num_experts=4)

        x = torch.randn(batch_size, d_model, requires_grad=True)
        out, _ = moe(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
