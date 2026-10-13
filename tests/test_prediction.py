"""Unit tests for Day 7: Module 9 (Prediction Heads) & Module 10 (Conformal Uncertainty)."""

import pytest
import numpy as np
import torch

from prediction.heads import (
    ReturnHead,
    DirectionHead,
    VolatilityHead,
    MultiTaskHead,
    MultiTaskLoss,
)
from prediction.conformal import (
    SplitConformal,
    AdaptiveConformal,
    compute_coverage,
)


# ── Module 9: Prediction Heads Tests ──────────────────────────────────────────

class TestPredictionHeads:

    def test_return_head_shape_and_grad(self):
        batch_size = 8
        d_model = 32
        horizon = 5

        head = ReturnHead(d_model=d_model, horizon=horizon)
        x = torch.randn(batch_size, d_model, requires_grad=True)
        out = head(x)

        assert out.shape == (batch_size, horizon)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_direction_head_shape_and_proba(self):
        batch_size = 8
        d_model = 32
        num_classes = 3

        head = DirectionHead(d_model=d_model, num_classes=num_classes)
        x = torch.randn(batch_size, d_model)
        logits = head(x)

        assert logits.shape == (batch_size, num_classes)
        probs = head.predict_proba(logits)
        assert probs.shape == (batch_size, num_classes)
        # Probabilities should sum to 1 across classes
        np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), 1.0, atol=1e-4)

    def test_volatility_head_non_negative(self):
        batch_size = 16
        d_model = 32
        horizon = 3

        head = VolatilityHead(d_model=d_model, horizon=horizon)
        # Test with extreme negative inputs
        x = torch.randn(batch_size, d_model) * 10 - 20
        out = head(x)

        assert out.shape == (batch_size, horizon)
        # Volatility must strictly be non-negative
        assert (out >= 0.0).all()

    def test_multitask_head_forward(self):
        batch_size = 4
        num_nodes = 6
        d_model = 64
        horizon = 5
        num_classes = 3

        mt_head = MultiTaskHead(
            d_model=d_model,
            horizon=horizon,
            num_classes=num_classes,
        )

        # Test on 3D input (batch, num_nodes, d_model)
        x = torch.randn(batch_size, num_nodes, d_model)
        outputs = mt_head(x)

        assert "return" in outputs
        assert "direction" in outputs
        assert "volatility" in outputs

        assert outputs["return"].shape == (batch_size, num_nodes, horizon)
        assert outputs["direction"].shape == (batch_size, num_nodes, num_classes)
        assert outputs["volatility"].shape == (batch_size, num_nodes, horizon)

    def test_multitask_loss_backward(self):
        batch_size = 8
        d_model = 32
        horizon = 1
        num_classes = 3

        mt_head = MultiTaskHead(d_model=d_model, horizon=horizon, num_classes=num_classes)
        criterion = MultiTaskLoss(alpha=1.0, beta=0.5, gamma=0.5)

        x = torch.randn(batch_size, d_model)
        preds = mt_head(x)

        targets = {
            "return": torch.randn(batch_size, horizon),
            "direction": torch.randint(0, num_classes, (batch_size,)),
            "volatility": torch.rand(batch_size, horizon) + 0.01,
        }

        total_loss, loss_dict = criterion(preds, targets)

        assert total_loss > 0
        assert "total" in loss_dict
        assert "return" in loss_dict
        assert "direction" in loss_dict
        assert "volatility" in loss_dict

        total_loss.backward()
        for param in mt_head.parameters():
            if param.requires_grad:
                assert param.grad is not None


# ── Module 10: Uncertainty Quantification Tests ───────────────────────────────

class TestConformalPrediction:

    def test_split_conformal_calibration_and_coverage(self):
        np.random.seed(42)
        n_cal = 500
        n_test = 200

        # Ground truth and noisy predictions
        y_true_cal = np.random.normal(loc=0.0, scale=1.0, size=n_cal)
        y_pred_cal = y_true_cal + np.random.normal(loc=0.0, scale=0.2, size=n_cal)

        y_true_test = np.random.normal(loc=0.0, scale=1.0, size=n_test)
        y_pred_test = y_true_test + np.random.normal(loc=0.0, scale=0.2, size=n_test)

        conformal = SplitConformal()
        conformal.calibrate(y_pred_cal, y_true_cal)

        alpha = 0.10  # 90% target coverage
        lower, upper = conformal.predict_interval(y_pred_test, alpha=alpha)

        assert lower.shape == y_pred_test.shape
        assert upper.shape == y_pred_test.shape
        assert (upper >= lower).all()

        metrics = compute_coverage(y_true_test, lower, upper)
        assert "coverage" in metrics
        assert "mean_width" in metrics
        assert "winkler_score" in metrics

        # Coverage on test set should be close to 90% (e.g. between 85% and 95%)
        assert 0.85 <= metrics["coverage"] <= 0.95

    def test_split_conformal_clipping_for_volatility(self):
        np.random.seed(42)
        y_cal = np.abs(np.random.normal(0.05, 0.02, 100))
        y_pred = y_cal + np.random.normal(0.0, 0.01, 100)

        conformal = SplitConformal()
        conformal.calibrate(y_pred, y_cal)

        lower, upper = conformal.predict_interval(y_pred[:10], alpha=0.1, clip_min=0.0)
        # Volatility lower bound must never be negative
        assert (lower >= 0.0).all()

    def test_adaptive_conformal_updates(self):
        np.random.seed(42)
        n_cal = 100
        y_cal = np.random.randn(n_cal)
        y_pred_cal = y_cal + 0.1 * np.random.randn(n_cal)

        aci = AdaptiveConformal(alpha=0.10, step_size=0.05)
        aci.calibrate(y_pred_cal, y_cal)

        initial_alpha = aci.current_alpha
        assert abs(initial_alpha - 0.10) < 1e-5

        # Case 1: Point is inside interval -> error = 0 -> alpha increases (narrowing interval)
        y_inside = 0.0
        lower, upper = aci.predict_interval(y_inside)
        new_alpha = aci.step(y_true=0.0, lower=lower, upper=upper)
        assert new_alpha > initial_alpha

        # Case 2: Extreme outlier outside interval -> error = 1 -> alpha decreases (widening interval)
        y_extreme_outlier = 1000.0
        lower2, upper2 = aci.predict_interval(0.0)
        alpha_after_miss = aci.step(y_true=y_extreme_outlier, lower=lower2, upper=upper2)
        assert alpha_after_miss < new_alpha

    def test_compute_coverage_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower = np.array([0.5, 1.5, 2.5, 3.5, 5.5])  # 4 out of 5 covered (80%)
        upper = np.array([1.5, 2.5, 3.5, 4.5, 6.5])

        metrics = compute_coverage(y_true, lower, upper)
        assert abs(metrics["coverage"] - 0.80) < 1e-4
        assert abs(metrics["mean_width"] - 1.0) < 1e-4
        assert metrics["winkler_score"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
