import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MacroEncoder(nn.Module):
    """
    Encoder for macroeconomic time-series data.
    """
    def __init__(self, input_dim: int, d_model: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4, # Default nhead
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor of shape (batch, d_model)
        """
        # Project input
        out = self.input_proj(x)
        
        # Pass through transformer (batch, seq_len, d_model)
        out = self.transformer(out)
        
        # Take the last timestep output as the macro embedding
        out = out[:, -1, :]
        
        return out
