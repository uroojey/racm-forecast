import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class FundamentalEncoder(nn.Module):
    """
    TabTransformer-style encoder for tabular fundamental data.
    """
    def __init__(self, input_dim: int, d_model: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project each feature (from dim 1 to d_model)
        self.input_proj = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_features)
        Returns:
            Tensor of shape (batch, d_model)
        """
        # Reshape to (batch, n_features, 1)
        x = x.unsqueeze(-1)
        
        # Project to d_model space: (batch, n_features, d_model)
        out = self.input_proj(x)
        
        # Transformer encoding
        out = self.transformer(out)
        
        # Mean pooling across the feature dimension
        out = out.mean(dim=1)
        
        return out
