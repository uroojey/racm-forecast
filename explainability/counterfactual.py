"""Module 11: Counterfactual Scenario Analysis for RACM-Forecast.

Provides pure PyTorch counterfactual perturbation generation and 'what-if' scenario
evaluations to quantify model sensitivity to economic and market events.
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn


class CounterfactualGenerator:
    """Generates counterfactual perturbations and scenario analyses for PyTorch models.

    Allows analysts to test 'what-if' market conditions (e.g., interest rate shock,
    volatility spike, or absent macro event) and assess how model predictions shift.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize CounterfactualGenerator.

        Args:
            model: PyTorch model to evaluate.
            feature_names: Optional list of human-readable feature names.
        """
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None

    def _resolve_feature_name(self, feature_idx: int) -> Union[str, int]:
        """Resolve feature index to human-readable string or integer identifier."""
        if self.feature_names is not None and 0 <= feature_idx < len(self.feature_names):
            return self.feature_names[feature_idx]
        return int(feature_idx)

    def _compute_delta(
        self,
        orig: Union[Dict[str, torch.Tensor], torch.Tensor],
        cf: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the prediction difference between counterfactual and original."""
        deltas: Dict[str, torch.Tensor] = {}

        if isinstance(orig, dict) and isinstance(cf, dict):
            for k in orig:
                if k in cf:
                    deltas[k] = cf[k] - orig[k]
            return deltas
        elif isinstance(orig, torch.Tensor) and isinstance(cf, torch.Tensor):
            diff = cf - orig
            return {
                "output": diff,
                "prediction": diff,
            }
        elif isinstance(orig, dict) and isinstance(cf, torch.Tensor):
            first_key = next(iter(orig.keys()))
            return {first_key: cf - orig[first_key]}
        else:
            diff = cf - orig
            return {
                "output": diff,
                "prediction": diff,
            }

    def _get_device(self) -> torch.device:
        """Get model device."""
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def generate(
        self,
        x: torch.Tensor,
        feature_idx: int,
        delta: float = 0.0,
        replace_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a single-feature counterfactual perturbation.

        Args:
            x: Input feature tensor of shape (..., num_features).
            feature_idx: Index of feature to perturb.
            delta: Additive perturbation amount to apply if replace_value is None.
            replace_value: Optional absolute replacement value for the feature.

        Returns:
            Dictionary containing:
                - 'original_prediction': Original model output.
                - 'counterfactual_prediction': Counterfactual model output.
                - 'delta': Dict showing output differences (counterfactual - original).
                - 'perturbed_feature': Name or index of the perturbed feature.
                - 'feature_idx': Integer index of perturbed feature.
                - 'perturbation': Delta value or replacement applied.
        """
        device = self._get_device()
        x_in = x.to(device)
        x_cf = x_in.clone()

        if replace_value is not None:
            x_cf[..., feature_idx] = float(replace_value)
            perturbation_info: Union[str, float] = f"replaced_with_{replace_value}"
        else:
            x_cf[..., feature_idx] = x_cf[..., feature_idx] + float(delta)
            perturbation_info = float(delta)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                orig_pred = self.model(x_in)
                cf_pred = self.model(x_cf)
        finally:
            if was_training:
                self.model.train()

        delta_dict = self._compute_delta(orig_pred, cf_pred)
        feature_label = self._resolve_feature_name(feature_idx)

        return {
            "original_prediction": orig_pred,
            "counterfactual_prediction": cf_pred,
            "delta": delta_dict,
            "perturbed_feature": feature_label,
            "feature_idx": int(feature_idx),
            "perturbation": perturbation_info,
        }

    def generate_batch(
        self,
        x: torch.Tensor,
        feature_indices: List[int],
        deltas: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run counterfactual perturbation analysis across multiple features.

        Args:
            x: Input feature tensor.
            feature_indices: List of feature indices to perturb individually.
            deltas: Optional list of perturbation magnitudes for each feature.
                Defaults to 1.0 for each feature if not specified.

        Returns:
            List of counterfactual result dictionaries.
        """
        if deltas is None:
            deltas_list = [1.0] * len(feature_indices)
        else:
            if len(deltas) != len(feature_indices):
                raise ValueError(
                    f"Length mismatch: {len(deltas)} deltas vs "
                    f"{len(feature_indices)} feature_indices."
                )
            deltas_list = [float(d) for d in deltas]

        results: List[Dict[str, Any]] = []
        for feat_idx, d in zip(feature_indices, deltas_list):
            res = self.generate(x, feature_idx=feat_idx, delta=d)
            results.append(res)

        return results

    def what_if_no_event(
        self,
        x: torch.Tensor,
        event_feature_indices: List[int],
    ) -> Dict[str, Any]:
        """Simulate a 'what-if-no-event' scenario by setting event features to zero.

        Evaluates the counterfactual scenario where an anticipated or past catalyst
        (e.g., FOMC rate hike, CPI surprise, geopolitical shock) did not take place.

        Args:
            x: Input feature tensor.
            event_feature_indices: List of indices corresponding to event indicators/embeddings.

        Returns:
            Dictionary containing original_prediction, counterfactual_prediction,
            delta difference, and event_features names/indices.
        """
        device = self._get_device()
        x_in = x.to(device)
        x_cf = x_in.clone()
        for idx in event_feature_indices:
            x_cf[..., idx] = 0.0

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                orig_pred = self.model(x_in)
                cf_pred = self.model(x_cf)
        finally:
            if was_training:
                self.model.train()

        delta_dict = self._compute_delta(orig_pred, cf_pred)
        event_names = [self._resolve_feature_name(idx) for idx in event_feature_indices]

        return {
            "original_prediction": orig_pred,
            "counterfactual_prediction": cf_pred,
            "delta": delta_dict,
            "event_features": event_names,
            "event_feature_indices": list(event_feature_indices),
        }

    def summary_report(
        self,
        x: torch.Tensor,
        feature_indices: List[int],
        deltas: Optional[List[float]] = None,
    ) -> str:
        """Generate a formatted multi-line string report of counterfactual scenarios.

        Args:
            x: Input feature tensor.
            feature_indices: Feature indices to evaluate.
            deltas: Optional perturbation deltas for each feature.

        Returns:
            Formatted scenario analysis report.
        """
        results = self.generate_batch(x, feature_indices=feature_indices, deltas=deltas)

        lines = [
            "=" * 72,
            "               Counterfactual Scenario Analysis Report",
            "=" * 72,
            f"{'Feature':<24} | {'Perturbation':<14} | {'Key Output Impact (Delta)':<28}",
            "-" * 72,
        ]

        for item in results:
            feat_name = str(item["perturbed_feature"])
            pert = str(item["perturbation"])
            delta_dict = item["delta"]

            # Summarize the impact deltas
            impact_strs = []
            for k, diff in delta_dict.items():
                if isinstance(diff, torch.Tensor):
                    if diff.numel() == 1:
                        val = float(diff.item())
                        impact_strs.append(f"{k}: {val:+.5f}")
                    else:
                        mean_val = float(diff.mean().item())
                        impact_strs.append(f"{k}(avg): {mean_val:+.5f}")
                else:
                    impact_strs.append(f"{k}: {diff}")

            impact_summary = ", ".join(impact_strs) if impact_strs else "No diff"
            lines.append(f"{feat_name:<24} | {pert:<14} | {impact_summary:<28}")

        lines.append("=" * 72)
        return "\n".join(lines)
