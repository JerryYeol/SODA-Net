"""
WaveFormer adapter for AGTCNet
Extract Wave2D block to replace standard Transformer in TCE module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Wave1D(nn.Module):
    """
    Adapted Wave equation operator for 1D EEG temporal sequences
    Based on Wave2D from WaveFormer, simplified for 1D case

    Wave equation: d2u/dt2 - c2(d2u/dx2) + αdu/dt = 0

    For 1D case:
    A(n) = DCT1D(φ(x))
    u(x, t) = IDCT1D(A(n) * (1 - (nπ/a)^2 * c2t2) * e^(-αt))
    """

    def __init__(self, dim=32, temporal_len=7, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.temporal_len = temporal_len

        # Depthwise convolution for initial processing
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # Linear projection
        self.linear = nn.Linear(dim, 2 * dim, bias=True)

        # Output layers
        self.out_norm = nn.LayerNorm(dim)
        self.out_linear = nn.Linear(dim, dim, bias=True)

        # Wave equation parameters
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.GELU(),
        )

        # Learnable wave speed and damping
        self.c = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    @staticmethod
    def get_cos_map(N, device=torch.device("cpu"), dtype=torch.float):
        """
        Generate DCT/IDCT cosine basis
        cos((x + 0.5) / N * n * π)
        """
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(N, device=torch.device("cpu"), dtype=torch.float):
        """
        Wave equation decay term: e^(-(nπ/a)^2)
        """
        weight_n = torch.linspace(0, torch.pi, N + 1, device=device, dtype=dtype)[:N]
        weight = torch.exp(-torch.pow(weight_n, 2))
        return weight

    def forward(self, x, freq_embed=None):
        """
        Args:
            x: (batch, temporal_len, dim) - temporal sequence
            freq_embed: optional frequency embedding

        Returns:
            x: (batch, temporal_len, dim) - transformed sequence
        """
        B, T, C = x.shape

        # Depthwise conv: (B, T, C) -> (B, C, T) -> (B, C, T) -> (B, T, C)
        x_conv = x.permute(0, 2, 1).contiguous()  # (B, C, T)
        x_conv = self.dwconv(x_conv)  # (B, C, T)
        x_conv = x_conv.permute(0, 2, 1).contiguous()  # (B, T, C)

        # Linear projection and split
        x_proj = self.linear(x_conv)  # (B, T, 2C)
        x, z = x_proj.chunk(chunks=2, dim=-1)  # Each (B, T, C)

        # Cache or compute DCT basis
        cached_weight = getattr(self, "__WEIGHT_COS__", None)
        if (T == getattr(self, "__T__", 0)) and cached_weight is not None and (cached_weight.device == x.device):
            weight_cos = cached_weight
        else:
            weight_cos = self.get_cos_map(T, device=x.device).detach_()
            setattr(self, "__T__", T)
            setattr(self, "__WEIGHT_COS__", weight_cos)

        # DCT: Transform to frequency domain
        # weight_cos: (T, T), x: (B, T, C)
        # We want: (B, C, T) @ (T, T) = (B, C, T)
        x_perm = x.permute(0, 2, 1).contiguous()  # (B, C, T)
        x_flat = x_perm.reshape(-1, 1, T)  # (B*C, 1, T)
        weight_kernel = weight_cos.view(T, 1, T)  # (T, 1, T)
        x_u0 = F.conv1d(x_flat, weight_kernel).squeeze(1)  # (B*C, T)
        x_u0 = x_u0.view(B, C, T).permute(0, 2, 1).contiguous()  # (B, T, C)

        # Velocity term (simplified as x_v0 ≈ x_u0 for initialization)
        x_v0 = x_u0

        # Frequency embedding (learnable time parameter)
        if freq_embed is None:
            freq_embed = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)

        t = self.to_k(freq_embed)
        c_t = self.c * t

        # Wave equation solution
        cos_term = torch.cos(c_t)
        eps = 1e-8
        sin_term = torch.sin(c_t) / (self.c + eps)

        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        final_term = wave_term + velocity_term

        # IDCT: Transform back to time domain
        x_perm = final_term.permute(0, 2, 1).contiguous()  # (B, C, T)
        x_flat = x_perm.reshape(-1, 1, T)  # (B*C, 1, T)
        weight_kernel_t = weight_cos.t().contiguous().view(T, 1, T)  # (T, 1, T)
        x_idct = F.conv1d(x_flat, weight_kernel_t).squeeze(1)  # (B*C, T)
        x_final = x_idct.view(B, C, T).permute(0, 2, 1).contiguous()  # (B, T, C)

        # Normalization and gating
        x_final = self.out_norm(x_final)
        gate = F.silu(z)
        x_gated = x_final * gate
        x_out = self.out_linear(x_gated)

        return x_out


class WaveFormerBlock(nn.Module):
    """
    WaveFormer block with residual connection
    Replaces standard Transformer attention + FFN
    """

    def __init__(self, dim=32, temporal_len=7, dropout=0.5, mlp_ratio=4.0):
        super().__init__()

        # Wave operator (replaces attention)
        self.norm1 = nn.LayerNorm(dim)
        self.wave = Wave1D(dim=dim, temporal_len=temporal_len, dropout=dropout)

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        # Frequency embedding (learnable)
        self.freq_embed = nn.Parameter(torch.zeros(temporal_len, dim))
        nn.init.trunc_normal_(self.freq_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (batch, temporal_len, dim)

        Returns:
            x: (batch, temporal_len, dim)
        """
        B = x.shape[0]

        # Expand freq_embed for batch
        freq_embed = self.freq_embed.unsqueeze(0).expand(B, -1, -1)

        # Wave operator with residual
        x = x + self.dropout(self.wave(self.norm1(x), freq_embed))

        # FFN with residual
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x


if __name__ == "__main__":
    # Test Wave1D
    batch_size = 8
    temporal_len = 7
    dim = 32

    x = torch.randn(batch_size, temporal_len, dim)

    wave = Wave1D(dim=dim, temporal_len=temporal_len)
    out = wave(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Test WaveFormerBlock
    block = WaveFormerBlock(dim=dim, temporal_len=temporal_len)
    out = block(x)

    print(f"WaveFormerBlock output shape: {out.shape}")
    print("✓ WaveFormer adapter test passed!")
