"""
Audio-Style Fusion Module (ASFM) — temporal audio fusion + style cross-attention.
Fusion formula: z^{i+1} = z^i + s_phi * SCA(c_s) + s_alpha * ACA(f_A^i)
Temporal window: f_A^i = concat(frame i-4 .. i+5), L=50 (10 frames * l=5).
"""

import torch
import torch.nn as nn


class AudioContextAggregation(nn.Module):
    """ACA: cross-attention over concatenated temporal audio window."""

    def __init__(self, dim: int, num_heads: int = 8,
                 audio_len: int = 5, window_frames: int = 10):
        super().__init__()
        self.audio_len = audio_len
        self.window_frames = window_frames
        self.total_len = audio_len * window_frames
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)

    def get_audio_window(self, audio_features, frame_idx):
        """
        Args:
            audio_features: (B, T, l, D) — per-frame audio features
            frame_idx: (B,) — current frame index per sample
        Returns:
            torch.Tensor: (B, L, D) temporal window, zero-padded at boundaries
        """
        B, T, l, D = audio_features.shape
        device = audio_features.device
        windows = []
        for b in range(B):
            idx = frame_idx[b].item()
            indices = list(range(idx - 4, idx + 6))  # [i-4, i+5] inclusive = 10 frames
            frames = []
            for fi in indices:
                if 0 <= fi < T:
                    frames.append(audio_features[b, fi])
                else:
                    frames.append(torch.zeros(l, D, device=device))
            windows.append(torch.cat(frames, dim=0))
        return torch.stack(windows, dim=0)

    def forward(self, hidden, audio_features, frame_idx):
        """
        Args:
            hidden: (B, L_q, D) — DiT query features
            audio_features: (B, T, l, D) — all-frame audio features
            frame_idx: (B,) — current frame index
        Returns:
            torch.Tensor: (B, L_q, D) aggregated audio context added to hidden states
        """
        kv = self.get_audio_window(audio_features, frame_idx)
        q = self.q_norm(hidden)
        kv = self.kv_norm(kv)
        q_t = q.permute(1, 0, 2)
        kv_t = kv.permute(1, 0, 2)
        out, _ = self.attn(q_t, kv_t, kv_t)
        return hidden + out.permute(1, 0, 2)


class StyleCrossAttention(nn.Module):
    """SCA: cross-attention with style embedding c_s as K, V."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)

    def forward(self, hidden, c_s):
        """
        Args:
            hidden: (B, L, D) — DiT hidden states
            c_s: (B, N_s, D) — style embedding from SEEM as K, V
        Returns:
            torch.Tensor: (B, L, D) style-attended features added to hidden states
        """
        q = self.q_norm(hidden)
        kv = self.kv_norm(c_s)
        q_t = q.permute(1, 0, 2)
        kv_t = kv.permute(1, 0, 2)
        out, _ = self.attn(q_t, kv_t, kv_t)
        return hidden + out.permute(1, 0, 2)


class AudioStyleFusionModule(nn.Module):
    """
    ASFM: combines ACA and SCA with layer-wise scaling factors.
    z^{i+1} = z^i + s_phi^i * SCA(c_s) + s_alpha^i * ACA(f_A^i)
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 audio_len: int = 5, window_frames: int = 10):
        super().__init__()
        self.aca = AudioContextAggregation(dim, num_heads, audio_len, window_frames)
        self.sca = StyleCrossAttention(dim, num_heads)

    def forward(self, hidden, c_s, audio_features, frame_idx, s_phi, s_alpha):
        """
        Args:
            hidden: (B, L, D) — DiT block output z^i
            c_s: (B, N_s, D) — style embedding from SEEM
            audio_features: (B, T, l, D) — audio features
            frame_idx: (B,) — current frame index
            s_phi: (B, d_b, 1) — style scaling factors from ScaleAdapter
            s_alpha: (B, d_b, 1) — audio scaling factors from ScaleAdapter
        Returns:
            torch.Tensor: (B, L, D) fused output z^{i+1}
        """
        sca_out = self.sca(hidden, c_s)
        aca_out = self.aca(hidden, audio_features, frame_idx)
        fused = hidden + s_phi * sca_out + s_alpha * aca_out
        return fused
