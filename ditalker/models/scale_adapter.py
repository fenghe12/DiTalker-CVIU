"""
Scale Adapter — Q-Former based layer-wise scaling factor predictor.
Paper Sec 3.2: "three-layer Q-Former architecture" (cite StyleCrafter, BLIP-2).

Q-Former core design:
1. Learned query tokens (d_b) + condition tokens cat(c_s, c_a) share self-attention
2. Cross-attention: only queries attend to condition features (queries=Q, conditions=K/V)  
3. Queries act as information bottleneck extracting per-layer scaling factors

Forward: s_phi, s_alpha = model(c_s, c_a)
"""

import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """
    Single Q-Former layer following BLIP-2 design:
    1. Shared self-attention: queries and condition tokens attend to each other
    2. Cross-attention: only queries (as Q) attend to condition features (as K/V)
    3. FFN on queries
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Step 1: Shared self-attention (queries + conditions concatenated)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Step 2: Cross-attention (only queries participate as Q)
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Step 3: FFN (only on queries)
        self.norm_ffn = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, queries, cond_tokens):
        """
        Args:
            queries:     (B, N_q, D) — learnable query tokens (d_b)
            cond_tokens: (B, N_c, D) — condition features cat(c_s, c_a)
        Returns:
            queries: (B, N_q, D) — updated queries
        """
        N_q = queries.shape[1]

        # Step 1: Shared self-attention
        # Concatenate queries and conditions, do self-attention together
        x = torch.cat([queries, cond_tokens], dim=1)  # (B, N_q+N_c, D)
        x_norm = self.norm_self(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]
        # Split back — only keep queries going forward
        queries = x[:, :N_q, :]

        # Step 2: Cross-attention (queries=Q, conditions=K/V)
        # This is the key Q-Former design: only queries interact with conditions
        q_norm = self.norm_cross_q(queries)
        kv_norm = self.norm_cross_kv(cond_tokens)
        queries = queries + self.cross_attn(q_norm, kv_norm, kv_norm)[0]

        # Step 3: FFN on queries only
        queries = queries + self.ffn(self.norm_ffn(queries))

        return queries


class ScaleAdapter(nn.Module):
    """
    Q-Former Scale Adapter (Paper Sec 3.2).
    
    d_b learnable query tokens serve as bottleneck, extracting per-layer
    scaling factors from shared condition features cat(c_a, c_s).
    
    Architecture: 3 QFormerLayers, each with:
      - Shared self-attention (queries ↔ conditions)
      - Cross-attention (queries → conditions, one-directional)
      - FFN (queries only)
    
    Output: s_phi, s_alpha via dual linear heads with Tanh.
    Init: zero-weight + unit-bias → tanh(1) ≈ 0.76 at start.
    """
    def __init__(self, feature_dim=1152, num_heads=8, num_layers=28, qformer_layers=3):
        super().__init__()
        self.num_layers = num_layers
        scale = feature_dim ** -0.5

        # d_b learnable query tokens (information bottleneck)
        self.query_tokens = nn.Parameter(torch.randn(1, num_layers, feature_dim) * scale)

        # Q-Former layers
        self.qformer_layers = nn.ModuleList([
            QFormerLayer(feature_dim, num_heads) for _ in range(qformer_layers)
        ])

        self.output_norm = nn.LayerNorm(feature_dim)

        # Dual output heads: Linear(D->32)->GELU->Linear(32->1)->Tanh
        self.style_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.audio_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        # Zero-init last linear + unit bias → initial output = tanh(1) ≈ 0.76
        nn.init.zeros_(self.style_head[2].weight)
        nn.init.ones_(self.style_head[2].bias)
        nn.init.zeros_(self.audio_head[2].weight)
        nn.init.ones_(self.audio_head[2].bias)

    def forward(self, c_s, c_a):
        """
        Args:
            c_s: (B, N_s, D) — style embedding from SEEM
            c_a: (B, N_a, D) — audio embedding (projected whisper features)
        Returns:
            s_phi:   (B, d_b, 1) — style scaling factors per DiT layer
            s_alpha: (B, d_b, 1) — audio scaling factors per DiT layer
        """
        B = c_s.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)  # (B, d_b, D)
        cond_tokens = torch.cat([c_s, c_a], dim=1)     # (B, N_s+N_a, D)

        # Pass through Q-Former layers
        for layer in self.qformer_layers:
            queries = layer(queries, cond_tokens)

        queries = self.output_norm(queries)

        # Dual heads
        s_phi = self.style_head(queries)      # (B, d_b, 1)
        s_alpha = self.audio_head(queries)    # (B, d_b, 1)
        return s_phi, s_alpha
