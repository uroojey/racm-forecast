"""Module 9: Prediction Heads for RACM-Forecast.

Implements specialized regression and classification prediction heads and a joint
multi-task training loss function:
- ReturnHead: Multi-horizon regression predicting asset returns.
- DirectionHead: 3-class classification predicting market direction (Down, Flat, Up).
- VolatilityHead: Non-negative regression predicting asset price volatility.
- MultiTaskHead: Joint architecture housing return, direction, and volatility heads.
- MultiTaskLoss: Joint loss combining return regression, direction classification,
  and volatility regression losses with customizable weighting.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReturnHead(nn.Module):
    """Regression head predicting asset price returns over a forecast horizon.

    Architecture:
        Linear(d_model, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, horizon)
    """

    def __init__(
        self,
        d_model: int,
        horizon: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ReturnHead.

        Args:
            d_model: Input representation dimension.
            horizon: Forecast horizon (number of return steps to predict).
            hidden_dim: Hidden dimension for intermediate MLP layer. Defaults to d_model.
            dropout: Dropout probability.
        """
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if hidden_dim is None:
            hidden_dim = d_model

        self.d_model = d_model
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor, squeeze: bool = False) -> torch.Tensor:
        """Forward pass predicting price returns.

        Args:
            x: Input tensor of shape (..., d_model).
            squeeze: If True and horizon == 1, squeezes the last dimension.

        Returns:
            Predicted returns of shape (..., horizon) or (...) if squeezed.
        """
        out = self.net(x)
        if squeeze and self.horizon == 1:
            return out.squeeze(-1)
        return out


class DirectionHead(nn.Module):
    """Classification head predicting market direction: Down (0), Flat (1), Up (2).

    Architecture:
        Linear(d_model, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize DirectionHead.

        Args:
            d_model: Input representation dimension.
            num_classes: Number of direction classes (typically 3: Down, Flat, Up).
            hidden_dim: Hidden dimension for intermediate MLP layer. Defaults to d_model.
            dropout: Dropout probability.
        """
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if hidden_dim is None:
            hidden_dim = d_model

        self.d_model = d_model
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting unnormalized direction logits.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Logits tensor of shape (..., num_classes).
        """
        return self.net(x)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert direction logits to normalized probability distribution.

        Args:
            logits: Logits tensor of shape (..., num_classes).

        Returns:
            Normalized class probabilities of shape (..., num_classes) summing to 1.
        """
        return F.softmax(logits, dim=-1)


class VolatilityHead(nn.Module):
    """Regression head predicting non-negative price volatility.

    Architecture:
        Linear(d_model, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, horizon) -> Softplus()
    """

    def __init__(
        self,
        d_model: int,
        horizon: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize VolatilityHead.

        Args:
            d_model: Input representation dimension.
            horizon: Forecast horizon (number of volatility steps to predict).
            hidden_dim: Hidden dimension for intermediate MLP layer. Defaults to d_model.
            dropout: Dropout probability.
        """
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if hidden_dim is None:
            hidden_dim = d_model

        self.d_model = d_model
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, squeeze: bool = False) -> torch.Tensor:
        """Forward pass predicting non-negative price volatility.

        Args:
            x: Input tensor of shape (..., d_model).
            squeeze: If True and horizon == 1, squeezes the last dimension.

        Returns:
            Predicted volatility tensor of shape (..., horizon) with values >= 0.
        """
        out = self.net(x)
        if squeeze and self.horizon == 1:
            return out.squeeze(-1)
        return out


class MultiTaskHead(nn.Module):
    """Multi-task prediction head combining return, direction, and volatility heads."""

    def __init__(
        self,
        d_model: int,
        horizon: int = 1,
        num_classes: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize MultiTaskHead.

        Args:
            d_model: Input representation dimension.
            horizon: Forecast horizon for return and volatility regression.
            num_classes: Number of classification targets for market direction.
            hidden_dim: Hidden dimension for all three sub-heads. Defaults to d_model.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.num_classes = num_classes

        self.return_head = ReturnHead(
            d_model=d_model,
            horizon=horizon,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.direction_head = DirectionHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.volatility_head = VolatilityHead(
            d_model=d_model,
            horizon=horizon,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all three prediction heads.

        Args:
            x: Input representation tensor of shape (..., d_model),
               such as (batch_size, d_model) or (batch_size, num_nodes, d_model).

        Returns:
            Dictionary containing:
                - 'return': torch.Tensor of shape (..., horizon)
                - 'direction': torch.Tensor logits of shape (..., num_classes)
                - 'volatility': torch.Tensor of shape (..., horizon)
        """
        return {
            "return": self.return_head(x),
            "direction": self.direction_head(x),
            "volatility": self.volatility_head(x),
        }

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Helper to compute direction probabilities via DirectionHead.

        Args:
            logits: Direction logits tensor of shape (..., num_classes).

        Returns:
            Normalized class probabilities of shape (..., num_classes).
        """
        return self.direction_head.predict_proba(logits)


class MultiTaskLoss(nn.Module):
    """Joint multi-task loss combining return, direction, and volatility losses:

        L_total = alpha * L_return + beta * L_direction + gamma * L_volatility
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        return_loss_type: str = "mse",
        volatility_loss_type: str = "mse",
    ) -> None:
        """Initialize MultiTaskLoss.

        Args:
            alpha: Weight for return regression loss.
            beta: Weight for direction classification loss.
            gamma: Weight for volatility regression loss.
            return_loss_type: Loss type for returns ('mse', 'smooth_l1', 'huber', 'mae', 'l1').
            volatility_loss_type: Loss type for volatility ('mse', 'smooth_l1', 'huber', 'mae', 'l1').
        """
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.return_loss_type = return_loss_type.lower()
        self.volatility_loss_type = volatility_loss_type.lower()

        self.return_criterion = self._get_regression_criterion(self.return_loss_type)
        self.direction_criterion = nn.CrossEntropyLoss()
        self.volatility_criterion = self._get_regression_criterion(self.volatility_loss_type)

    @staticmethod
    def _get_regression_criterion(loss_type: str) -> nn.Module:
        """Instantiate regression loss module by string identifier."""
        loss_type = loss_type.lower()
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type in ("smooth_l1", "smoothl1", "huber"):
            return nn.SmoothL1Loss()
        elif loss_type in ("mae", "l1"):
            return nn.L1Loss()
        else:
            raise ValueError(
                f"Unsupported regression loss type '{loss_type}'. "
                f"Supported types: 'mse', 'smooth_l1', 'huber', 'mae', 'l1'."
            )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted joint multi-task loss and per-task losses.

        Handles inputs of shape (batch, horizon) or (batch, num_nodes, horizon).

        Args:
            predictions: Dict containing 'return', 'direction', 'volatility'.
            targets: Dict containing ground truth for 'return', 'direction', 'volatility'.

        Returns:
            Tuple of:
                - total_loss: Scalar torch.Tensor combining all weighted losses.
                - loss_dict: Dict with keys 'total', 'return', 'direction', 'volatility'.
        """
        if "return" not in predictions or "return" not in targets:
            raise KeyError("Both predictions and targets must contain 'return' key.")
        if "direction" not in predictions or "direction" not in targets:
            raise KeyError("Both predictions and targets must contain 'direction' key.")
        if "volatility" not in predictions or "volatility" not in targets:
            raise KeyError("Both predictions and targets must contain 'volatility' key.")

        pred_ret = predictions["return"]
        target_ret = targets["return"]

        pred_dir = predictions["direction"]
        target_dir = targets["direction"]

        pred_vol = predictions["volatility"]
        target_vol = targets["volatility"]

        # ── 1. Return Regression Loss ──────────────────────────────────────────
        target_ret = target_ret.to(dtype=pred_ret.dtype, device=pred_ret.device)
        if pred_ret.shape != target_ret.shape and pred_ret.numel() == target_ret.numel():
            target_ret = target_ret.view_as(pred_ret)
        loss_ret = self.return_criterion(pred_ret, target_ret)

        # ── 2. Direction Classification Loss ──────────────────────────────────
        num_classes = pred_dir.shape[-1]
        pred_dir_flat = pred_dir.reshape(-1, num_classes)

        if target_dir.shape == pred_dir.shape and target_dir.is_floating_point():
            # Soft targets (class probabilities)
            target_dir_flat = target_dir.reshape(-1, num_classes).to(
                dtype=pred_dir.dtype, device=pred_dir.device
            )
        else:
            # Discrete class indices
            if target_dir.ndim == pred_dir.ndim and target_dir.shape[-1] == 1:
                target_dir = target_dir.squeeze(-1)
            target_dir_flat = target_dir.reshape(-1).long().to(device=pred_dir.device)

        loss_dir = self.direction_criterion(pred_dir_flat, target_dir_flat)

        # ── 3. Volatility Regression Loss ──────────────────────────────────────
        target_vol = target_vol.to(dtype=pred_vol.dtype, device=pred_vol.device)
        if pred_vol.shape != target_vol.shape and pred_vol.numel() == target_vol.numel():
            target_vol = target_vol.view_as(pred_vol)
        loss_vol = self.volatility_criterion(pred_vol, target_vol)

        # ── 4. Total Multi-Task Loss ───────────────────────────────────────────
        total_loss = (
            self.alpha * loss_ret
            + self.beta * loss_dir
            + self.gamma * loss_vol
        )

        loss_dict = {
            "total": total_loss,
            "return": loss_ret,
            "direction": loss_dir,
            "volatility": loss_vol,
        }

        return total_loss, loss_dict
