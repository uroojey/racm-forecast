"""Module 11: Explainability & Risk Analysis for RACM-Forecast.

Implements model interpretability, feature importance attribution, graph-level explanations,
and counterfactual 'what-if' risk scenario analysis:
- SHAPExplainer: KernelSHAP feature attribution for PyTorch prediction heads.
- GraphExplainer: Node- and edge-level graph explanations with GNNExplainer/gradient fallback.
- CounterfactualGenerator: Counterfactual perturbations and event impact scenarios.
"""

from explainability.shap_explainer import SHAPExplainer
from explainability.graph_explainer import GraphExplainer
from explainability.counterfactual import CounterfactualGenerator

__all__ = [
    "SHAPExplainer",
    "GraphExplainer",
    "CounterfactualGenerator",
]
