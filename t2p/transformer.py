from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        mlp_dim_factor: int,
        cross_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_dim_factor = mlp_dim_factor
        self.cross_embed_dim = cross_embed_dim
        # Self-attention layer
        self.qkv_layer_norm = nn.LayerNorm(self.embed_dim)
        self.causal_self_attention = CausalSelfAttention(
            self.embed_dim,
            self.num_heads,
            self.dropout,
        )
        # Cross-attention layer
        self.q_layer_norm = nn.LayerNorm(self.embed_dim)
        self.kv_layer_norm = nn.LayerNorm(cross_embed_dim or self.embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            dropout=self.dropout,
            kdim=cross_embed_dim,
            vdim=cross_embed_dim,
            batch_first=True,
        )
        # MLP layer
        self.mlp_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(
                self.embed_dim,
                self.mlp_dim_factor * self.embed_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.mlp_dim_factor * self.embed_dim,
                self.embed_dim,
            ),
            nn.Dropout(self.dropout),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        cross_embeddings: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.qkv_layer_norm(embeddings)
        residuals = self.causal_self_attention(qkv, padding_mask=padding_mask)
        embeddings = embeddings + residuals

        if cross_embeddings is not None:
            q = self.q_layer_norm(embeddings)
            kv = self.kv_layer_norm(cross_embeddings)
            residuals = self.cross_attention(
                query=q,
                key=kv,
                value=kv,
                need_weights=False,
            )[0]
            embeddings = embeddings + residuals

        x = self.mlp_layer_norm(embeddings)
        residuals = self.mlp(x)
        embeddings = embeddings + residuals

        return embeddings


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"{embed_dim=} should be divisible by {num_heads=}"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, H = x.size()  # batch size, sequence length, embed_dim

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Attention mask is True when attending, False when not attending
        attn_mask = torch.ones((B, T, T), dtype=torch.bool, device=x.device).tril(0)
        # Combine causal mask with padding mask
        if padding_mask is not None:
            mask = torch.vmap(lambda m: torch.outer(m, m))(padding_mask)
            # Set True in the diagonal
            mask[:, torch.arange(T), torch.arange(T)] = True
            attn_mask = attn_mask.logical_and(mask)
        attn_mask = (
            attn_mask.type(q.dtype).masked_fill(
                ~attn_mask,
                torch.finfo(q.dtype).min,
            )
            - 1
        )
        if attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None, :, :]  # For multi-head attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, H)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
