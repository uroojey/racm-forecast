"""Integration tests for RACM-Forecast End-to-End Pipeline (Day 8 Part 2)."""

import sys
from pathlib import Path
from typing import Dict

import pytest
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline import RACMForecastPipeline


class TestRACMPipeline:
    """Integration test suite for RACMForecastPipeline."""

    def test_pipeline_construction(self) -> None:
        """Test that the pipeline builds without errors from custom config."""
        custom_config = {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.05,
            "input_dim": 16,
            "num_nodes": 8,
            "n_regimes": 3,
            "num_experts": 3,
            "horizon": 3,
            "num_classes": 3,
            "alpha": 0.05,
            "sequence_length": 40,
        }
        pipeline = RACMForecastPipeline(custom_config)

        # Must subclass nn.Module
        assert isinstance(pipeline, nn.Module)
        assert pipeline.d_model == 64
        assert pipeline.n_heads == 4
        assert pipeline.n_layers == 2
        assert pipeline.input_dim == 16
        assert pipeline.num_nodes == 8
        assert pipeline.n_regimes == 3
        assert pipeline.num_experts == 3
        assert pipeline.horizon == 3

        # Submodules must be proper nn.Modules
        assert hasattr(pipeline, "temporal_encoder")
        assert isinstance(pipeline.temporal_encoder, nn.Module)
        assert hasattr(pipeline, "graph_transformer")
        assert isinstance(pipeline.graph_transformer, nn.Module)
        assert hasattr(pipeline, "moe_fusion")
        assert isinstance(pipeline.moe_fusion, nn.Module)
        assert hasattr(pipeline, "prediction_heads")
        assert isinstance(pipeline.prediction_heads, nn.Module)
        assert hasattr(pipeline, "loss_fn")
        assert isinstance(pipeline.loss_fn, nn.Module)

        # Auxiliary components
        assert hasattr(pipeline, "causal_learner")
        assert hasattr(pipeline, "regime_detector")
        assert hasattr(pipeline, "conformal")

    def test_from_default_config(self) -> None:
        """Test factory method creating pipeline with standard defaults."""
        pipeline = RACMForecastPipeline.from_default_config()

        assert isinstance(pipeline, RACMForecastPipeline)
        assert pipeline.d_model == 128
        assert pipeline.n_heads == 8
        assert pipeline.n_layers == 4
        assert pipeline.dropout == 0.1
        assert pipeline.input_dim == 20
        assert pipeline.num_nodes == 10
        assert pipeline.n_regimes == 4
        assert pipeline.num_experts == 4
        assert pipeline.horizon == 5
        assert pipeline.num_classes == 3
        assert pipeline.alpha == 0.1
        assert pipeline.sequence_length == 60

    def test_forward_pass(self) -> None:
        """Test a complete end-to-end forward pass with dummy data."""
        pipeline = RACMForecastPipeline.from_default_config()
        pipeline.eval()
        batch_size = 4
        seq_len = 60
        input_dim = 20
        num_nodes = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        causal_adj = torch.rand(num_nodes, num_nodes)
        regime_probs = torch.softmax(torch.randn(batch_size, num_nodes, 4), dim=-1)

        with torch.no_grad():
            outputs = pipeline(x, causal_adj, regime_probs)

        # Core predictions
        assert "return" in outputs
        assert "direction" in outputs
        assert "volatility" in outputs
        assert outputs["return"].shape == (batch_size, num_nodes, 5)  # horizon=5
        assert outputs["direction"].shape == (batch_size, num_nodes, 3)  # 3 classes
        assert outputs["volatility"].shape == (batch_size, num_nodes, 5)

        # Intermediate representations
        assert "temporal_encoding" in outputs
        assert outputs["temporal_encoding"].shape == (batch_size, seq_len, 128)
        assert "graph_encoding" in outputs
        assert outputs["graph_encoding"].shape == (batch_size, num_nodes, 128)
        assert "moe_output" in outputs
        assert outputs["moe_output"].shape == (batch_size, num_nodes, 128)
        assert "gate_weights" in outputs
        assert outputs["gate_weights"].shape == (batch_size, num_nodes, 4)

        # Volatility non-negativity check
        assert (outputs["volatility"] >= 0.0).all()

        # Gate weights sum to 1 check
        gate_sum = outputs["gate_weights"].sum(dim=-1)
        torch.testing.assert_close(gate_sum, torch.ones_like(gate_sum), atol=1e-5, rtol=1e-5)

    def test_forward_pass_no_regime_probs(self) -> None:
        """Test forward pass when regime_probs is None (unconditional gating)."""
        pipeline = RACMForecastPipeline.from_default_config()
        pipeline.eval()
        batch_size = 2
        seq_len = 60
        input_dim = 20
        num_nodes = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        causal_adj = torch.rand(num_nodes, num_nodes)

        with torch.no_grad():
            outputs = pipeline(x, causal_adj, regime_probs=None)

        assert outputs["return"].shape == (batch_size, num_nodes, 5)
        assert outputs["direction"].shape == (batch_size, num_nodes, 3)
        assert outputs["volatility"].shape == (batch_size, num_nodes, 5)
        assert outputs["gate_weights"].shape == (batch_size, num_nodes, 4)

    def test_forward_pass_batched_adjacency(self) -> None:
        """Test forward pass with per-batch 3D causal adjacency matrix."""
        pipeline = RACMForecastPipeline.from_default_config()
        pipeline.eval()
        batch_size = 3
        seq_len = 60
        input_dim = 20
        num_nodes = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        causal_adj_batched = torch.rand(batch_size, num_nodes, num_nodes)
        regime_probs = torch.softmax(torch.randn(batch_size, num_nodes, 4), dim=-1)

        with torch.no_grad():
            outputs = pipeline(x, causal_adj_batched, regime_probs)

        assert outputs["return"].shape == (batch_size, num_nodes, 5)
        assert outputs["direction"].shape == (batch_size, num_nodes, 3)
        assert outputs["volatility"].shape == (batch_size, num_nodes, 5)

    def test_forward_pass_4d_input(self) -> None:
        """Test forward pass with 4D per-node temporal inputs."""
        pipeline = RACMForecastPipeline.from_default_config()
        pipeline.eval()
        batch_size = 2
        num_nodes = 10
        seq_len = 60
        input_dim = 20

        x_4d = torch.randn(batch_size, num_nodes, seq_len, input_dim)
        causal_adj = torch.rand(num_nodes, num_nodes)

        with torch.no_grad():
            outputs = pipeline(x_4d, causal_adj)

        assert outputs["return"].shape == (batch_size, num_nodes, 5)
        assert outputs["direction"].shape == (batch_size, num_nodes, 3)
        assert outputs["volatility"].shape == (batch_size, num_nodes, 5)

    def test_backward_pass(self) -> None:
        """Test that gradients flow through the entire pipeline back to input and parameters."""
        pipeline = RACMForecastPipeline.from_default_config()
        pipeline.train()
        batch_size = 2
        seq_len = 30
        input_dim = 20
        num_nodes = 10

        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        causal_adj = torch.rand(num_nodes, num_nodes)
        regime_probs = torch.softmax(torch.randn(batch_size, num_nodes, 4), dim=-1)

        outputs = pipeline(x, causal_adj, regime_probs)

        # Joint scalar objective for gradient test
        loss = (
            outputs["return"].sum()
            + outputs["direction"].sum()
            + outputs["volatility"].sum()
        )
        loss.backward()

        # Check input gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

        # Check key submodule parameters have gradients
        temporal_params = list(pipeline.temporal_encoder.parameters())
        assert any(p.grad is not None for p in temporal_params)

        graph_params = list(pipeline.graph_transformer.parameters())
        assert any(p.grad is not None for p in graph_params)

        moe_params = list(pipeline.moe_fusion.parameters())
        assert any(p.grad is not None for p in moe_params)

        head_params = list(pipeline.prediction_heads.parameters())
        assert any(p.grad is not None for p in head_params)

    def test_loss_computation(self) -> None:
        """Test multi-task loss computation and gradient backpropagation."""
        pipeline = RACMForecastPipeline.from_default_config()
        pipeline.train()
        batch_size = 4
        seq_len = 60
        input_dim = 20
        num_nodes = 10
        horizon = 5

        x = torch.randn(batch_size, seq_len, input_dim)
        causal_adj = torch.rand(num_nodes, num_nodes)
        regime_probs = torch.softmax(torch.randn(batch_size, num_nodes, 4), dim=-1)

        predictions = pipeline(x, causal_adj, regime_probs)

        # Synthetic multi-task ground truth targets
        targets: Dict[str, torch.Tensor] = {
            "return": torch.randn(batch_size, num_nodes, horizon),
            "direction": torch.randint(0, 3, (batch_size, num_nodes)),
            "volatility": torch.rand(batch_size, num_nodes, horizon) + 0.01,
        }

        total_loss, loss_dict = pipeline.compute_loss(predictions, targets)

        # Assert loss structure
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # Scalar
        assert total_loss.item() > 0.0

        for key in ["total", "return", "direction", "volatility"]:
            assert key in loss_dict
            assert isinstance(loss_dict[key], torch.Tensor)
            assert loss_dict[key].item() >= 0.0

        # Gradient flow from multi-task loss
        pipeline.zero_grad()
        total_loss.backward()
        temporal_grad = next(pipeline.temporal_encoder.parameters()).grad
        assert temporal_grad is not None

    def test_parameter_count(self) -> None:
        """Verify the model has a reasonable number of parameters organized by module."""
        pipeline = RACMForecastPipeline.from_default_config()

        # Check get_parameters structure
        params_by_module = pipeline.get_parameters()
        assert "temporal_encoder" in params_by_module
        assert "graph_transformer" in params_by_module
        assert "moe_fusion" in params_by_module
        assert "prediction_heads" in params_by_module

        for mod_name, param_list in params_by_module.items():
            assert isinstance(param_list, list)
            assert len(param_list) > 0, f"Module {mod_name} has no parameters"

        # Check total parameter count
        total_params = pipeline.count_parameters(trainable_only=True)
        # Standard configuration should have ~1M to ~10M parameters
        assert 100_000 < total_params < 20_000_000, f"Unexpected parameter count: {total_params}"

    def test_uncertainty_quantification_integration(self) -> None:
        """Test SplitConformal calibration and interval construction through pipeline."""
        pipeline = RACMForecastPipeline.from_default_config()
        pipeline.eval()
        batch_size = 2
        seq_len = 60
        input_dim = 20
        num_nodes = 10
        horizon = 5

        # Synthetic calibration set
        val_pred = np.random.randn(50, horizon)
        val_true = val_pred + np.random.normal(0, 0.02, size=val_pred.shape)

        pipeline.calibrate_uncertainty(val_pred, val_true)
        assert pipeline.conformal.scores_ is not None
        assert pipeline.conformal.n_calibration_ == 50 * horizon

        x = torch.randn(batch_size, seq_len, input_dim)
        causal_adj = torch.rand(num_nodes, num_nodes)

        with torch.no_grad():
            out = pipeline.predict_with_uncertainty(x, causal_adj, alpha=0.1)

        assert "return_lower" in out
        assert "return_upper" in out
        assert out["return_lower"].shape == (batch_size, num_nodes, horizon)
        assert out["return_upper"].shape == (batch_size, num_nodes, horizon)
        assert (out["return_lower"] <= out["return_upper"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
