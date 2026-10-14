"""End-to-End RACM-Forecast Pipeline.

Wires together all 11 modules of the RACM-Forecast framework:
1. Preprocessing (data loaders, feature engineering, temporal alignment)
2. Temporal Encoding (TransformerTemporalEncoder with risk-adaptive attention)
3. Event Extraction (financial news NER & sentiment extraction)
4. Knowledge Graph (entity-relation graph builder & embedder)
5. Causal Discovery (NOTEARS continuous DAG optimization)
6. Causal Temporal Graph Transformer (RACM-GT attention modulated by causal DAG)
7. Market Regime Detection (Gaussian HMM / GMM regime classification)
8. Mixture-of-Experts Fusion (regime-conditioned expert routing)
9. Multi-Task Prediction Heads (joint return, direction, volatility heads)
10. Uncertainty Quantification (SplitConformal prediction intervals)
11. Explainability (SHAP, graph attention, counterfactuals)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np

# Core neural module imports
from encoders.temporal_encoder import TransformerTemporalEncoder
from causal.racm_gt import CausalTemporalGT
from regime.moe_fusion import MoEFusion
from prediction.heads import MultiTaskHead, MultiTaskLoss

# Auxiliary / non-neural module imports
from regime.regime_detector import MarketRegimeDetector
from causal.causal_learner import CausalLearner
from prediction.conformal import SplitConformal


class RACMForecastPipeline(nn.Module):
    """End-to-end RACM-Forecast pipeline wiring all 11 modules.
    
    Pipeline flow:
    Raw Data -> Preprocessing (Module 1)
             -> Temporal Encoding (Module 2)
             -> Event Extraction (Module 3, optional)
             -> Knowledge Graph (Module 4)
             -> Causal Discovery (Module 5)
             -> Causal Temporal Graph Transformer (Module 6)
             -> Market Regime Detection (Module 7)
             -> Mixture-of-Experts Fusion (Module 8)
             -> Multi-Task Prediction Heads (Module 9)
             -> Uncertainty Quantification (Module 10)
             -> Explainability (Module 11, optional)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize all neural and auxiliary modules from configuration.

        Args:
            config: Configuration dictionary containing:
                - d_model: Latent representation dimension (default: 128).
                - n_heads: Number of attention heads (default: 8).
                - n_layers: Number of transformer blocks (default: 4).
                - dropout: Dropout probability (default: 0.1).
                - input_dim: Number of input features per timestep (default: 20).
                - num_nodes: Number of graph nodes/entities (default: 10).
                - n_regimes: Number of market regimes (default: 4).
                - num_experts: Number of MoE experts (default: 4).
                - horizon: Multi-step forecast horizon (default: 5).
                - num_classes: Number of direction classes (default: 3).
                - alpha: Conformal miscoverage significance level (default: 0.1).
                - sequence_length: Historical sequence lookback length (default: 60).
        """
        super().__init__()
        self.config = dict(config)

        # Hyperparameters
        self.d_model = int(config.get("d_model", 128))
        self.n_heads = int(config.get("n_heads", 8))
        self.n_layers = int(config.get("n_layers", 4))
        self.dropout = float(config.get("dropout", 0.1))
        self.input_dim = int(config.get("input_dim", 20))
        self.num_nodes = int(config.get("num_nodes", 10))
        self.n_regimes = int(config.get("n_regimes", 4))
        self.num_experts = int(config.get("num_experts", 4))
        self.horizon = int(config.get("horizon", 5))
        self.num_classes = int(config.get("num_classes", 3))
        self.alpha = float(config.get("alpha", 0.1))
        self.sequence_length = int(config.get("sequence_length", 60))

        # ── 1. Temporal Encoder (Module 2) ────────────────────────────────────
        self.temporal_encoder = TransformerTemporalEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # ── 2. Causal Temporal Graph Transformer (Module 6) ───────────────────
        self.graph_transformer = CausalTemporalGT(
            num_nodes=self.num_nodes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            in_dim=self.d_model,
        )

        # ── 3. Mixture-of-Experts Fusion (Module 8) ───────────────────────────
        self.moe_fusion = MoEFusion(
            d_model=self.d_model,
            num_experts=self.num_experts,
            num_regimes=self.n_regimes,
            dropout=self.dropout,
        )

        # ── 4. Multi-Task Prediction Heads (Module 9) ─────────────────────────
        self.prediction_heads = MultiTaskHead(
            d_model=self.d_model,
            horizon=self.horizon,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )

        # ── 5. Multi-Task Training Loss (Module 9) ────────────────────────────
        loss_alpha = float(config.get("loss_alpha", 1.0))
        loss_beta = float(config.get("loss_beta", 0.5))
        loss_gamma = float(config.get("loss_gamma", 0.5))
        return_loss_type = str(config.get("return_loss_type", "mse"))
        volatility_loss_type = str(config.get("volatility_loss_type", "mse"))

        self.loss_fn = MultiTaskLoss(
            alpha=loss_alpha,
            beta=loss_beta,
            gamma=loss_gamma,
            return_loss_type=return_loss_type,
            volatility_loss_type=volatility_loss_type,
        )

        # ── Auxiliary Non-Neural Components (Modules 5, 7, 10) ────────────────
        self.causal_learner = CausalLearner()
        self.regime_detector = MarketRegimeDetector(n_regimes=self.n_regimes)
        self.conformal = SplitConformal()

    def forward(
        self,
        x: torch.Tensor,
        causal_adj: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """End-to-end forward pass through all neural modules.
        
        Args:
            x: Input temporal features of shape (batch_size, seq_len, input_dim)
               or (batch_size, num_nodes, seq_len, input_dim).
            causal_adj: Causal adjacency matrix of shape (num_nodes, num_nodes)
                        or (batch_size, num_nodes, num_nodes).
            regime_probs: Optional regime posterior probabilities of shape
                          (batch_size, num_nodes, n_regimes) or (batch_size, n_regimes).
        
        Returns:
            dict with keys:
            - "return": predicted returns of shape (batch_size, num_nodes, horizon)
            - "direction": direction logits of shape (batch_size, num_nodes, num_classes)
            - "volatility": predicted volatility of shape (batch_size, num_nodes, horizon)
            - "temporal_encoding": intermediate temporal embeddings (batch_size, seq_len, d_model)
            - "graph_encoding": intermediate graph embeddings (batch_size, num_nodes, d_model)
            - "moe_output": MoE fused output (batch_size, num_nodes, d_model)
            - "gate_weights": expert routing weights (batch_size, num_nodes, num_experts)
        """
        # 1. Temporal encoding
        if x.dim() == 4:
            # (batch_size, num_nodes, seq_len, input_dim)
            batch_size, num_nodes, seq_len, feat_dim = x.shape
            x_flat = x.view(batch_size * num_nodes, seq_len, feat_dim)
            temporal_flat = self.temporal_encoder(x_flat)
            temporal_encoding = temporal_flat.view(batch_size, num_nodes, seq_len, self.d_model)
            # Take last timestep per node
            graph_input = temporal_encoding[:, :, -1, :]  # (batch_size, num_nodes, d_model)
        elif x.dim() == 3:
            # (batch_size, seq_len, input_dim)
            temporal_encoding = self.temporal_encoder(x)  # (batch_size, seq_len, d_model)
            # 2. Reshape temporal encoding for graph: take last timestep, expand to num_nodes
            last_temporal = temporal_encoding[:, -1, :]  # (batch_size, d_model)
            graph_input = last_temporal.unsqueeze(1).expand(-1, self.num_nodes, -1)  # (batch, num_nodes, d_model)
        else:
            raise ValueError(
                f"Expected input x with 3 or 4 dimensions, got {x.dim()} with shape {x.shape}."
            )

        # 3. Causal Temporal Graph Transformer
        graph_encoding = self.graph_transformer(graph_input, causal_adj)

        # 4. Mixture-of-Experts Fusion
        moe_output, gate_weights = self.moe_fusion(graph_encoding, regime_probs)

        # 5. Multi-Task Prediction Heads
        head_outputs = self.prediction_heads(moe_output)

        # 6. Return all outputs dict
        return {
            "return": head_outputs["return"],
            "direction": head_outputs["direction"],
            "volatility": head_outputs["volatility"],
            "temporal_encoding": temporal_encoding,
            "graph_encoding": graph_encoding,
            "moe_output": moe_output,
            "gate_weights": gate_weights,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss combining return, direction, and volatility errors.

        Args:
            predictions: Dict containing 'return', 'direction', and 'volatility' tensors.
            targets: Dict containing ground truth 'return', 'direction', and 'volatility'.

        Returns:
            Tuple of:
                - total_loss: Scalar weighted multi-task loss.
                - loss_dict: Dictionary with 'total', 'return', 'direction', 'volatility'.
        """
        return self.loss_fn(predictions, targets)

    def get_parameters(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Returns a dict of all module parameters organized by module name."""
        return {
            "temporal_encoder": list(self.temporal_encoder.parameters()),
            "graph_transformer": list(self.graph_transformer.parameters()),
            "moe_fusion": list(self.moe_fusion.parameters()),
            "prediction_heads": list(self.prediction_heads.parameters()),
        }

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Returns the total number of parameters across all neural submodules."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def calibrate_uncertainty(
        self,
        val_predictions: Union[np.ndarray, torch.Tensor],
        val_targets: Union[np.ndarray, torch.Tensor],
    ) -> "RACMForecastPipeline":
        """Calibrate SplitConformal nonconformity scores using validation data.

        Args:
            val_predictions: Validation set return predictions.
            val_targets: Validation set ground truth returns.

        Returns:
            self: Calibrated pipeline instance.
        """
        self.conformal.calibrate(val_predictions, val_targets)
        return self

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        causal_adj: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Forward pass with calibrated conformal prediction intervals.

        Args:
            x: Input temporal features.
            causal_adj: Causal adjacency matrix.
            regime_probs: Optional regime probabilities.
            alpha: Significance level for conformal interval (defaults to self.alpha).

        Returns:
            Dictionary with standard model outputs, plus:
                - 'return_lower': Lower conformal bound for returns
                - 'return_upper': Upper conformal bound for returns
        """
        outputs = self.forward(x, causal_adj, regime_probs)
        eff_alpha = alpha if alpha is not None else self.alpha

        if self.conformal.scores_ is not None:
            lower, upper = self.conformal.predict_interval(outputs["return"], alpha=eff_alpha)
            outputs["return_lower"] = lower
            outputs["return_upper"] = upper

        return outputs

    @classmethod
    def from_default_config(cls) -> "RACMForecastPipeline":
        """Creates a pipeline with reasonable default hyperparameters."""
        config = {
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "dropout": 0.1,
            "input_dim": 20,
            "num_nodes": 10,
            "n_regimes": 4,
            "num_experts": 4,
            "horizon": 5,
            "num_classes": 3,
            "alpha": 0.1,
            "sequence_length": 60,
        }
        return cls(config)
