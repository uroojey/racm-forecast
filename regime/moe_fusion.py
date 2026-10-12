"""Regime-Conditioned Mixture-of-Experts (MoE) Fusion Module for RACM-Forecast.

Implements specialized expert networks routed dynamically by a regime-aware
gating mechanism conditioned on market state beliefs.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertNetwork(nn.Module):
    """Specialized MLP expert network for market representation processing."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the ExpertNetwork.

        Args:
            d_model: Dimension of input and output representations.
            hidden_dim: Dimension of intermediate MLP layer (defaults to 4 * d_model).
            dropout: Dropout probability.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for expert network.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).
        """
        h = self.fc1(x)
        h = self.ln(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        out = self.drop2(h)
        return out


class RegimeGatingNetwork(nn.Module):
    """Gating network that routes representations conditioned on regime probabilities."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        num_regimes: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the RegimeGatingNetwork.

        Args:
            d_model: Dimension of representation features.
            num_experts: Number of expert networks to route across.
            num_regimes: Number of regime states in posterior probability vector.
            dropout: Dropout probability (optional).
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_regimes = num_regimes

        self.gate_no_regime = nn.Linear(d_model, num_experts)
        self.gate_with_regime = nn.Linear(d_model + num_regimes, num_experts)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute routing probabilities across experts.

        Args:
            x: Node/temporal features of shape (batch_size, ..., d_model).
            regime_probs: Optional regime probabilities of shape
                (batch_size, ..., num_regimes).

        Returns:
            Softmax gating probabilities of shape (batch_size, ..., num_experts).
        """
        if regime_probs is not None:
            # Handle broadcasting when regime_probs has fewer dimensions than x
            if regime_probs.dim() < x.dim():
                diff = x.dim() - regime_probs.dim()
                for _ in range(diff):
                    regime_probs = regime_probs.unsqueeze(1)
                expand_shape = list(x.shape[:-1]) + [regime_probs.shape[-1]]
                regime_probs = regime_probs.expand(*expand_shape)

            fused_input = torch.cat([x, regime_probs], dim=-1)
            gate_logits = self.gate_with_regime(self.dropout(fused_input))
        else:
            gate_logits = self.gate_no_regime(self.dropout(x))

        return F.softmax(gate_logits, dim=-1)


class MoEFusion(nn.Module):
    """Regime-Conditioned Mixture-of-Experts (MoE) Fusion Layer."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        num_regimes: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize MoEFusion.

        Args:
            d_model: Model feature dimension.
            num_experts: Number of expert networks.
            num_regimes: Number of semantic regimes (defaults to num_experts).
            hidden_dim: Hidden dimension of each expert (defaults to 4 * d_model).
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_regimes = num_regimes if num_regimes is not None else num_experts

        self.experts = nn.ModuleList([
            ExpertNetwork(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.gating_network = RegimeGatingNetwork(
            d_model=d_model,
            num_experts=num_experts,
            num_regimes=self.num_regimes,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        regime_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for MoE fusion.

        Args:
            x: Input tensor of shape (batch_size, ..., d_model).
            regime_probs: Optional tensor of shape (batch_size, ..., num_regimes).

        Returns:
            Tuple of:
                - out: Fused representation with residual connection and LayerNorm
                  of shape (batch_size, ..., d_model).
                - gates: Expert routing weights of shape (batch_size, ..., num_experts).
        """
        gates = self.gating_network(x, regime_probs)

        # Run input through all experts and compute weighted sum
        expert_outputs = [expert(x) for expert in self.experts]
        fused = sum(gates[..., i : i + 1] * expert_outputs[i] for i in range(self.num_experts))

        # Residual connection and layer normalization
        out = self.layer_norm(x + fused)
        return out, gates
