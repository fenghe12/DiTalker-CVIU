"""
Emotion Cross-Attention (ECA) — single cross-attention layer.
c_emo = c_text + c_ref serves as K, V to inject emotion context into DiT hidden states.
"""

import torch
import torch.nn as nn


class EmotionCrossAttention(nn.Module):
    """Single cross-attention layer with emotion context c_emo as K, V."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)

    def forward(self, hidden, c_emo):
        """
        Args:
            hidden: (B, L, D) — DiT hidden states
            c_emo: (B, N_emo, D) — emotion context (c_text + c_ref) as K, V
        Returns:
            torch.Tensor: (B, L, D) attended features added to hidden states
        """
        q = self.q_norm(hidden)
        kv = self.kv_norm(c_emo)
        q_t = q.permute(1, 0, 2)
        kv_t = kv.permute(1, 0, 2)
        out, _ = self.attn(q_t, kv_t, kv_t)
        return hidden + out.permute(1, 0, 2)
