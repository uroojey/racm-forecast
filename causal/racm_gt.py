"""Causal Temporal Graph Transformer (RACM-GT) in PyTorch."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGraphAttention(nn.Module):
    """Multi-head attention where attention scores are modulated by causal graph structure.
    
    Given node features X in R^{(B, N, D)} and causal adjacency matrix A in R^{(N, N)} or R^{(B, N, N)},
    computes attention scores modulated by A, applies softmax, and computes attended value features.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable scale for causal adjacency bias
        self.adj_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for causal temporal graph attention.
        
        Args:
            x: Node representations of shape (batch_size, num_nodes, d_model)
               or (num_nodes, d_model).
            causal_adj: Causal adjacency matrix of shape (num_nodes, num_nodes)
                        or (batch_size, num_nodes, num_nodes).

        Returns:
            Updated node representations of shape (batch_size, num_nodes, d_model).
        """
        # Handle 2D input (num_nodes, d_model) -> (1, num_nodes, d_model)
        has_batch = (x.dim() == 3)
        if not has_batch:
            x = x.unsqueeze(0)

        batch_size, num_nodes, d_model = x.shape
        residual = x

        # Ensure causal_adj has shape (batch_size, 1, num_nodes, num_nodes)
        if causal_adj.dim() == 2:
            adj = causal_adj.unsqueeze(0).unsqueeze(1)  # (1, 1, N, N)
        elif causal_adj.dim() == 3:
            adj = causal_adj.unsqueeze(1)  # (B, 1, N, N)
        else:
            raise ValueError(f"Unsupported causal_adj shape: {causal_adj.shape}")

        # Linear projections & split heads: (B, N, d_model) -> (B, n_heads, N, d_k)
        q = self.q_proj(x).view(batch_size, num_nodes, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: (B, n_heads, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Modulate scores with causal structure
        scores = scores + self.adj_scale * adj

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context: (B, n_heads, N, d_k)
        context = torch.matmul(attn_weights, v)

        # Concatenate heads: (B, N, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, d_model)
        out = self.out_proj(context)
        out = self.dropout(out)

        # Residual connection + LayerNorm
        out = self.norm(residual + out)

        if not has_batch:
            out = out.squeeze(0)

        return out


class GraphTransformerBlock(nn.Module):
    """Transformer block with Causal Temporal Graph Attention and Position-wise FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.attn = TemporalGraphAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through attention + FFN with residual connections."""
        attn_out = self.attn(x, causal_adj)
        ffn_out = self.ffn(attn_out)
        return self.norm(attn_out + ffn_out)


class CausalTemporalGT(nn.Module):
    """Causal Temporal Graph Transformer (RACM-GT) stacking multiple transformer blocks.
    
    Args:
        num_nodes: Total number of graph nodes/entities.
        d_model: Latent dimension for node representations.
        n_heads: Number of attention heads.
        n_layers: Number of stacked GraphTransformerBlocks.
        dropout: Dropout rate.
        in_dim: Optional input feature dimension. If specified and different from d_model,
                projects input from in_dim to d_model.
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        in_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.in_dim = in_dim if in_dim is not None else d_model

        if self.in_dim != d_model:
            self.input_proj = nn.Linear(self.in_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.layers = nn.ModuleList([
            GraphTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        causal_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Processes node features through causal graph transformer blocks.
        
        Args:
            x: Input tensor of shape (batch_size, num_nodes, in_dim).
            causal_adj: Causal adjacency matrix of shape (num_nodes, num_nodes)
                        or (batch_size, num_nodes, num_nodes).

        Returns:
            Updated embeddings of shape (batch_size, num_nodes, d_model).
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, causal_adj)
        return h
