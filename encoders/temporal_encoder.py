import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        # x is (batch, seq_len, d_model). pe is (max_len, 1, d_model).
        # We need to reshape pe to match or transpose.
        # usually pe is (max_len, 1, d_model), if we want to add to (batch, seq_len, d_model), 
        # we can transpose pe to (1, max_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, 0, :].unsqueeze(0)
        return self.dropout(x)


class RiskAdaptiveAttention(nn.Module):
    """
    Custom multi-head attention that modulates attention weights based on recent volatility.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        
        # Volatility gate MLP
        self.volatility_gate = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_heads)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            volatility: (batch, 1) scalar volatility estimate
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # (batch, n_heads, seq_len, d_k)
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        attn_weights = F.softmax(scores, dim=-1)

        # Gate computation
        # gate: (batch, n_heads)
        gate_logits = self.volatility_gate(volatility)
        gate = torch.sigmoid(gate_logits).view(batch_size, self.n_heads, 1, 1)

        # Modulate attention weights
        attn_weights = attn_weights * gate
        # Re-normalize just in case, though the prompt says "multiply attention weights by gate"
        
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(out)


class TransformerEncoderBlock(nn.Module):
    """
    Encoder block using RiskAdaptiveAttention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = RiskAdaptiveAttention(d_model, n_heads, dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            volatility: (batch, 1)
        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x, volatility)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerTemporalEncoder(nn.Module):
    """
    Transformer-based temporal encoder for time-series forecasting.
    """
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # Volatility estimator
        self.vol_estimator = nn.Sequential(
            nn.Linear(input_dim * 10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Project input
        out = self.input_proj(x)
        out = self.pe(out)
        
        # Estimate volatility from last 10 timesteps of raw input
        # Assuming sequence length >= 10. If not, pad it.
        if seq_len >= 10:
            last_10 = x[:, -10:, :].reshape(batch_size, -1)
        else:
            pad = torch.zeros(batch_size, 10 - seq_len, input_dim, device=x.device)
            last_10 = torch.cat([pad, x], dim=1).reshape(batch_size, -1)
            
        volatility = self.vol_estimator(last_10) # (batch, 1)
        
        # Pass through layers
        for layer in self.layers:
            out = layer(out, volatility)
            
        return out


class TCNBlock(nn.Module):
    """
    Temporal Convolutional block.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        
        # Causal padding is applied manually in forward, so padding=0 here
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation
        ))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation
        ))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len)
        """
        # Apply causal padding
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNTemporalEncoder(nn.Module):
    """
    Stack of TCNBlocks with exponentially increasing dilation.
    """
    def __init__(self, input_dim: int, d_model: int, n_layers: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(d_model, d_model, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, d_model)
        """
        out = self.input_proj(x)
        # Transpose to (batch, channels, seq_len) for conv1d
        out = out.transpose(1, 2)
        out = self.tcn(out)
        # Transpose back
        out = out.transpose(1, 2)
        return out
