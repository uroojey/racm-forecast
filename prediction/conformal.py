"""Module 10: Conformal Prediction & Uncertainty Quantification for RACM-Forecast.

Implements distribution-free conformal prediction and adaptive calibration methods:
- SplitConformal: Inductive split conformal prediction providing distribution-free
  finite-sample coverage guarantees for regression tasks.
- AdaptiveConformal: Online Adaptive Conformal Inference (ACI) that dynamically updates
  the nominal coverage level in response to empirical coverage errors, coping with
  market regime shifts and non-stationarity.
- compute_coverage: Comprehensive evaluation utility calculating empirical coverage,
  mean interval width, and Winkler interval score.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class SplitConformal:
    """Distribution-free split conformal prediction for regression intervals.

    Uses calibration set residuals R_i = |y_i - y_hat_i| to compute prediction
    intervals with exact (1 - alpha) coverage under exchangeability.
    """

    def __init__(self) -> None:
        """Initialize SplitConformal predictor."""
        self.scores_: Optional[np.ndarray] = None
        self.n_calibration_: int = 0

    def calibrate(
        self,
        y_pred: Union[np.ndarray, torch.Tensor],
        y_true: Union[np.ndarray, torch.Tensor],
    ) -> "SplitConformal":
        """Compute and store nonconformity scores on a hold-out calibration set.

        Args:
            y_pred: Model predictions on calibration set.
            y_true: Ground truth targets on calibration set.

        Returns:
            self: The calibrated SplitConformal instance.
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        y_pred_arr = np.asarray(y_pred, dtype=float)
        y_true_arr = np.asarray(y_true, dtype=float)

        if y_pred_arr.shape != y_true_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_pred shape {y_pred_arr.shape} "
                f"does not match y_true shape {y_true_arr.shape}."
            )

        residuals = np.abs(y_true_arr - y_pred_arr).ravel()
        if residuals.size == 0:
            raise ValueError("Calibration set must contain at least one sample.")

        self.scores_ = np.sort(residuals)
        self.n_calibration_ = len(self.scores_)
        return self

    def get_quantile(self, alpha: float = 0.1) -> float:
        """Compute conformal quantile with finite-sample correction.

        Uses the order statistic index ceil((n + 1) * (1 - alpha)).

        Args:
            alpha: Miscoverage rate (e.g. 0.1 for 90% coverage).

        Returns:
            Conformal nonconformity quantile score q.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
        if self.scores_ is None or self.n_calibration_ == 0:
            raise RuntimeError("SplitConformal must be calibrated before querying quantiles.")

        n = self.n_calibration_
        # Finite-sample correction: index ceil((n + 1) * (1 - alpha))
        k = math.ceil((n + 1) * (1.0 - alpha))
        idx = min(max(k - 1, 0), n - 1)
        return float(self.scores_[idx])

    def predict_interval(
        self,
        y_pred: Union[float, np.ndarray, torch.Tensor],
        alpha: float = 0.1,
        clip_min: Optional[float] = None,
    ) -> Tuple[Any, Any]:
        """Construct prediction interval [lower_bound, upper_bound] at nominal 1 - alpha level.

        Supports scalar floats, numpy arrays, and torch tensors of arbitrary dimension.

        Args:
            y_pred: Point prediction(s) to construct intervals around.
            alpha: Miscoverage level (e.g. 0.1 for 90% coverage).
            clip_min: Optional lower bound cutoff (e.g. 0.0 for volatility).

        Returns:
            Tuple of (lower_bound, upper_bound) matching the type and shape of y_pred.
        """
        q = self.get_quantile(alpha)

        if isinstance(y_pred, torch.Tensor):
            lower = y_pred - q
            upper = y_pred + q
            if clip_min is not None:
                lower = torch.clamp(lower, min=clip_min)
                upper = torch.maximum(upper, lower)
            return lower, upper

        if isinstance(y_pred, (int, float, np.number)) or np.ndim(y_pred) == 0:
            val = float(y_pred)
            lower_val = val - q
            upper_val = val + q
            if clip_min is not None:
                lower_val = max(lower_val, clip_min)
                upper_val = max(upper_val, lower_val)
            return lower_val, upper_val

        y_arr = np.asarray(y_pred, dtype=float)
        lower_arr = y_arr - q
        upper_arr = y_arr + q
        if clip_min is not None:
            lower_arr = np.maximum(lower_arr, clip_min)
            upper_arr = np.maximum(upper_arr, lower_arr)
        return lower_arr, upper_arr


class AdaptiveConformal:
    """Online Adaptive Conformal Inference (ACI).

    Dynamically adjusts the nominal error level alpha_t to maintain empirical target
    coverage under distribution shift and changing market regimes:
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    where err_t = 1 if y_t not in [lower_t, upper_t] else 0.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        step_size: float = 0.01,
    ) -> None:
        """Initialize AdaptiveConformal predictor.

        Args:
            alpha: Target nominal miscoverage rate (e.g. 0.1 for 90% coverage).
            step_size: Learning rate / step size gamma controlling adaptation speed.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
        if step_size <= 0.0:
            raise ValueError(f"step_size must be positive, got {step_size}")

        self.target_alpha = float(alpha)
        self.step_size = float(step_size)
        self.current_alpha = float(alpha)
        self.conformal = SplitConformal()

    def calibrate(
        self,
        y_pred_cal: Union[np.ndarray, torch.Tensor],
        y_true_cal: Union[np.ndarray, torch.Tensor],
    ) -> "AdaptiveConformal":
        """Calibrate underlying conformal nonconformity scores and reset alpha_t.

        Args:
            y_pred_cal: Calibration set model predictions.
            y_true_cal: Calibration set ground truth targets.

        Returns:
            self: The calibrated AdaptiveConformal instance.
        """
        self.conformal.calibrate(y_pred_cal, y_true_cal)
        self.current_alpha = self.target_alpha
        return self

    def predict_interval(
        self,
        y_pred: Union[float, np.ndarray, torch.Tensor],
        clip_min: Optional[float] = None,
    ) -> Tuple[Any, Any]:
        """Predict interval using current adaptive miscoverage level alpha_t.

        Args:
            y_pred: Point prediction(s).
            clip_min: Optional lower bound cutoff (e.g. 0.0 for volatility).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        return self.conformal.predict_interval(
            y_pred,
            alpha=self.current_alpha,
            clip_min=clip_min,
        )

    def step(
        self,
        y_true: Union[float, np.ndarray, torch.Tensor],
        lower: Union[float, np.ndarray, torch.Tensor],
        upper: Union[float, np.ndarray, torch.Tensor],
    ) -> float:
        """Observe realized target y_t, evaluate coverage error, and update alpha_t.

        Update rule:
            alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        where err_t = 1 if y_t not in [lower, upper] else 0.
        Maintains alpha_{t+1} in (0, 1).

        Args:
            y_true: Realized ground truth value(s).
            lower: Interval lower bound.
            upper: Interval upper bound.

        Returns:
            Updated alpha_{t+1}.
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(lower, torch.Tensor):
            lower = lower.detach().cpu().numpy()
        if isinstance(upper, torch.Tensor):
            upper = upper.detach().cpu().numpy()

        if np.ndim(y_true) > 0:
            y_arr = np.asarray(y_true, dtype=float)
            l_arr = np.asarray(lower, dtype=float)
            u_arr = np.asarray(upper, dtype=float)
            err_t = float(np.mean((y_arr < l_arr) | (y_arr > u_arr)))
        else:
            y_val = float(y_true)
            l_val = float(lower)
            u_val = float(upper)
            err_t = 1.0 if (y_val < l_val or y_val > u_val) else 0.0

        alpha_next = self.current_alpha + self.step_size * (self.target_alpha - err_t)

        # Keep alpha strictly in (0, 1)
        eps = 1e-4
        self.current_alpha = float(max(eps, min(1.0 - eps, float(alpha_next))))
        return self.current_alpha


def compute_coverage(
    y_true: Union[np.ndarray, torch.Tensor],
    lower: Union[np.ndarray, torch.Tensor],
    upper: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.1,
) -> Dict[str, float]:
    """Compute empirical coverage, average interval width, and Winkler interval score.

    The Winkler score evaluates interval quality by penalizing width and violations:
        Score = (U - L) + (2 / alpha) * (L - y) * I(y < L) + (2 / alpha) * (y - U) * I(y > U)

    Args:
        y_true: Ground truth target values.
        lower: Predicted lower bounds.
        upper: Predicted upper bounds.
        alpha: Nominal miscoverage rate (default 0.1 for 90% target coverage).

    Returns:
        Dictionary containing:
            - 'coverage': Proportion of targets covered within [lower, upper].
            - 'mean_width': Average prediction interval width (upper - lower).
            - 'winkler_score': Winkler interval score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(lower, torch.Tensor):
        lower = lower.detach().cpu().numpy()
    if isinstance(upper, torch.Tensor):
        upper = upper.detach().cpu().numpy()

    y_arr = np.asarray(y_true, dtype=float)
    l_arr = np.asarray(lower, dtype=float)
    u_arr = np.asarray(upper, dtype=float)

    if y_arr.shape != l_arr.shape or l_arr.shape != u_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_arr.shape}, lower {l_arr.shape}, "
            f"upper {u_arr.shape} must have identical shapes."
        )
    if y_arr.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Coverage
    covered = (y_arr >= l_arr) & (y_arr <= u_arr)
    coverage = float(np.mean(covered))

    # Mean width
    width = u_arr - l_arr
    mean_width = float(np.mean(width))

    # Winkler score
    below = l_arr - y_arr
    above = y_arr - u_arr
    penalty = np.where(
        below > 0,
        (2.0 / alpha) * below,
        np.where(above > 0, (2.0 / alpha) * above, 0.0),
    )
    winkler_score = float(np.mean(width + penalty))

    return {
        "coverage": coverage,
        "mean_width": mean_width,
        "winkler_score": winkler_score,
    }
