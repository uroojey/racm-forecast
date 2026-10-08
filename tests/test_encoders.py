"""Unit tests for the encoders module (Module 2)."""
import pytest
import torch
import torch.nn as nn


# ── Config constants matching default.yaml ────
BATCH = 4
SEQ_LEN = 60
INPUT_DIM = 20       # number of market features
FUND_DIM = 9         # fundamental features (revenue, eps, etc.)
MACRO_DIM = 4        # GDP, inflation, interest_rate, unemployment
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1


# ── Temporal Encoder Tests ────────────────────

class TestTransformerTemporalEncoder:
    """Tests for the Transformer-based temporal encoder."""

    def test_output_shape(self):
        from encoders.temporal_encoder import TransformerTemporalEncoder
        model = TransformerTemporalEncoder(
            input_dim=INPUT_DIM, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = model(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL), \
            f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"

    def test_different_input_dims(self):
        from encoders.temporal_encoder import TransformerTemporalEncoder
        for dim in [5, 10, 50]:
            model = TransformerTemporalEncoder(
                input_dim=dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1
            )
            x = torch.randn(2, 30, dim)
            out = model(x)
            assert out.shape == (2, 30, 64)

    def test_gradients_flow(self):
        from encoders.temporal_encoder import TransformerTemporalEncoder
        model = TransformerTemporalEncoder(
            input_dim=INPUT_DIM, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=2, dropout=0.0
        )
        x = torch.randn(2, 30, INPUT_DIM, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow back to input"


class TestTCNTemporalEncoder:
    """Tests for the TCN-based temporal encoder."""

    def test_output_shape(self):
        from encoders.temporal_encoder import TCNTemporalEncoder
        model = TCNTemporalEncoder(
            input_dim=INPUT_DIM, d_model=D_MODEL,
            n_layers=N_LAYERS, kernel_size=3, dropout=DROPOUT
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = model(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL), \
            f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"

    def test_causal_no_future_leak(self):
        """Verify TCN is causal: changing future inputs shouldn't affect past outputs."""
        from encoders.temporal_encoder import TCNTemporalEncoder
        model = TCNTemporalEncoder(
            input_dim=5, d_model=32, n_layers=3, kernel_size=3, dropout=0.0
        )
        model.eval()

        x = torch.randn(1, 20, 5)
        out1 = model(x)

        # Modify the last 5 timesteps
        x_modified = x.clone()
        x_modified[:, 15:, :] = torch.randn(1, 5, 5)
        out2 = model(x_modified)

        # First 10 timesteps should be identical (causality)
        # Note: with dilation, receptive field grows, so check early timesteps
        assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5), \
            "TCN should be causal — early outputs must not depend on future inputs"


class TestRiskAdaptiveAttention:
    """Tests for the risk-adaptive attention mechanism."""

    def test_output_shape(self):
        from encoders.temporal_encoder import RiskAdaptiveAttention
        attn = RiskAdaptiveAttention(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0)
        q = k = v = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        vol = torch.randn(BATCH, 1).abs()  # volatility is non-negative
        out = attn(q, k, v, vol)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_volatility_affects_output(self):
        """Higher volatility should produce different outputs than lower."""
        from encoders.temporal_encoder import RiskAdaptiveAttention
        attn = RiskAdaptiveAttention(d_model=64, n_heads=4, dropout=0.0)
        attn.eval()
        q = k = v = torch.randn(1, 10, 64)
        out_low = attn(q, k, v, torch.tensor([[0.01]]))
        out_high = attn(q, k, v, torch.tensor([[10.0]]))
        assert not torch.allclose(out_low, out_high, atol=1e-4), \
            "Different volatility levels should produce different outputs"


# ── Fundamental Encoder Tests ─────────────────

class TestFundamentalEncoder:
    """Tests for the fundamental data encoder."""

    def test_output_shape(self):
        from encoders.fundamental_encoder import FundamentalEncoder
        model = FundamentalEncoder(input_dim=FUND_DIM, d_model=D_MODEL)
        x = torch.randn(BATCH, FUND_DIM)
        out = model(x)
        assert out.shape == (BATCH, D_MODEL), \
            f"Expected ({BATCH}, {D_MODEL}), got {out.shape}"


# ── Macro Encoder Tests ──────────────────────

class TestMacroEncoder:
    """Tests for the macro encoder."""

    def test_output_shape(self):
        from encoders.macro_encoder import MacroEncoder
        model = MacroEncoder(input_dim=MACRO_DIM, d_model=D_MODEL)
        x = torch.randn(BATCH, SEQ_LEN, MACRO_DIM)
        out = model(x)
        assert out.shape == (BATCH, D_MODEL), \
            f"Expected ({BATCH}, {D_MODEL}), got {out.shape}"

    def test_single_timestep(self):
        from encoders.macro_encoder import MacroEncoder
        model = MacroEncoder(input_dim=MACRO_DIM, d_model=64)
        x = torch.randn(2, 1, MACRO_DIM)
        out = model(x)
        assert out.shape == (2, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
