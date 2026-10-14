"""Module 11: SHAP-based Feature Importance for RACM-Forecast.

Wraps PyTorch prediction models (e.g. MultiTaskHead, ReturnHead, etc.) to compute
model-agnostic feature attributions using SHAP (SHapley Additive exPlanations).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False


class SHAPExplainer:
    """SHAP-based feature attribution explainer for PyTorch prediction heads.

    Wraps any PyTorch prediction model (e.g., ReturnHead, MultiTaskHead, or custom models)
    and uses SHAP KernelExplainer to compute feature importance scores and rankings.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the SHAPExplainer.

        Args:
            model: PyTorch model to explain. Must accept torch.Tensor inputs.
            feature_names: Optional list of human-readable feature names corresponding
                to input feature dimensions.
        """
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None

    def _check_shap_available(self) -> None:
        """Ensure the shap package is installed.

        Raises:
            ImportError: If the shap package is not available.
        """
        if not _SHAP_AVAILABLE or shap is None:
            raise ImportError(
                "The 'shap' package is required for SHAPExplainer but is not installed. "
                "Please install it using 'pip install shap'."
            )

    def _get_device(self) -> torch.device:
        """Determine the PyTorch device of the underlying model."""
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def _model_predict(self, x: np.ndarray) -> np.ndarray:
        """Forward wrapper for SHAP KernelExplainer.

        Converts numpy input to a PyTorch tensor, executes the model in eval mode
        with torch.no_grad, extracts the regression target ('return' if output is a dict,
        or the raw output otherwise), and converts back to a numpy array.

        Args:
            x: Input feature array of shape (batch_size, n_features).

        Returns:
            Numpy array of predictions with shape (batch_size,) or (batch_size, horizon).
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        device = self._get_device()
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=device)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(x_tensor)
                if isinstance(out, dict):
                    if "return" in out:
                        out = out["return"]
                    else:
                        out = next(iter(out.values()))

                if isinstance(out, torch.Tensor):
                    out_np = out.detach().cpu().numpy()
                else:
                    out_np = np.asarray(out)
        finally:
            if was_training:
                self.model.train()

        # If 2D with single target column (N, 1), squeeze to 1D (N,) for KernelExplainer
        if out_np.ndim == 2 and out_np.shape[1] == 1:
            out_np = out_np.reshape(-1)
        elif out_np.ndim == 0:
            out_np = out_np.reshape(1)

        return out_np

    def explain(self, X: np.ndarray, n_background: int = 100) -> Dict[str, Any]:
        """Compute SHAP values using KernelExplainer.

        Takes background data from the first n_background samples of X and computes
        SHAP values across all provided samples in X.

        Args:
            X: Input samples of shape (n_samples, n_features).
            n_background: Number of samples from X to use as background baseline.

        Returns:
            Dictionary containing:
                - 'shap_values': np.ndarray of computed SHAP attribution values.
                - 'base_value': float or np.ndarray expected value of the baseline.
                - 'feature_names': List of feature names.

        Raises:
            ImportError: If shap is not installed.
            ValueError: If X is not a 2D array or contains 0 samples.
        """
        self._check_shap_available()

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array of shape (n_samples, n_features), got shape {X_arr.shape}")
        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        n_samples, n_features = X_arr.shape

        # Resolve feature names
        if self.feature_names is not None:
            if len(self.feature_names) == n_features:
                names = list(self.feature_names)
            elif len(self.feature_names) < n_features:
                names = list(self.feature_names) + [
                    f"feature_{i}" for i in range(len(self.feature_names), n_features)
                ]
            else:
                names = list(self.feature_names[:n_features])
        else:
            names = [f"feature_{i}" for i in range(n_features)]

        # Prepare background baseline
        n_bg = min(max(1, n_background), n_samples)
        background = X_arr[:n_bg]

        # Initialize KernelExplainer and calculate SHAP values
        explainer = shap.KernelExplainer(self._model_predict, background)
        shap_values = explainer.shap_values(X_arr)

        # Normalize shap_values representation
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                shap_values_arr = np.asarray(shap_values[0])
            else:
                shap_values_arr = np.asarray(shap_values)
        else:
            shap_values_arr = np.asarray(shap_values)

        # Normalize base_value
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            if len(base_value) == 1:
                base_value = base_value[0]
            else:
                base_value = np.asarray(base_value)

        if isinstance(base_value, np.ndarray) and base_value.size == 1:
            base_value = float(base_value.item())
        elif isinstance(base_value, (int, float, np.floating)):
            base_value = float(base_value)

        return {
            "shap_values": shap_values_arr,
            "base_value": base_value,
            "feature_names": names,
        }

    def explain_top_k(
        self,
        X: np.ndarray,
        k: int = 5,
        n_background: int = 100,
    ) -> List[Dict[str, Any]]:
        """Rank and return top-k most important features by mean absolute SHAP value.

        Args:
            X: Input samples of shape (n_samples, n_features).
            k: Number of top features to return.
            n_background: Number of background samples.

        Returns:
            List of dictionaries, each formatted as:
                {'feature': str, 'importance': float, 'rank': int}

        Raises:
            ImportError: If shap is not installed.
        """
        explanation = self.explain(X, n_background=n_background)
        shap_vals = explanation["shap_values"]
        feature_names = explanation["feature_names"]

        # Aggregate absolute SHAP values across samples (and horizons if multi-dimensional)
        if shap_vals.ndim == 2:
            # (n_samples, n_features)
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
        elif shap_vals.ndim == 3:
            # Check which axis corresponds to features
            if shap_vals.shape[-1] == len(feature_names):
                mean_abs = np.mean(np.abs(shap_vals), axis=(0, 1))
            else:
                mean_abs = np.mean(np.abs(shap_vals), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(shap_vals), axis=0).ravel()

        ranking = np.argsort(-mean_abs)
        k_clamped = min(max(1, k), len(feature_names))

        top_k: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranking[:k_clamped], start=1):
            top_k.append({
                "feature": str(feature_names[idx]),
                "importance": float(mean_abs[idx]),
                "rank": rank,
            })

        return top_k

    def summary_report(self, X: np.ndarray, n_background: int = 100) -> str:
        """Generate a formatted multi-line string report of top features and importance scores.

        Args:
            X: Input samples of shape (n_samples, n_features).
            n_background: Number of background samples.

        Returns:
            Formatted string report.

        Raises:
            ImportError: If shap is not installed.
        """
        X_arr = np.asarray(X)
        n_features = X_arr.shape[1] if X_arr.ndim == 2 else 5
        k = min(10, n_features)
        top_k = self.explain_top_k(X, k=k, n_background=n_background)

        lines = [
            "=" * 64,
            "            SHAP Feature Importance Summary Report",
            "=" * 64,
            f"Total Evaluated Samples : {len(X_arr)}",
            f"Background Samples Used : {min(n_background, len(X_arr))}",
            "-" * 64,
            f"{'Rank':<6} | {'Feature':<28} | {'Mean |SHAP| Importance':<22}",
            "-" * 64,
        ]

        for item in top_k:
            lines.append(
                f"{item['rank']:<6} | {item['feature']:<28} | {item['importance']:<22.6f}"
            )

        lines.append("=" * 64)
        return "\n".join(lines)
