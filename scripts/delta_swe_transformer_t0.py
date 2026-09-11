"""T0: ForeSWE-referenced factorized spatiotemporal Transformer for the 7-day
-> 1-day Sierra SWE-change task. See docs/ForeSWE_T0_Architecture_Mapping.md
for the full architecture derivation. Plain spatial attention only -- no
geography bias (that is T1, not implemented here).
"""
from __future__ import annotations

import math

import torch
from torch import nn

N_HISTORY_DAYS = 7
N_CHANNELS = 12
GRID_H = 120
GRID_W = 240
PATCH = 8
GRID_H_P = GRID_H // PATCH  # 15
GRID_W_P = GRID_W // PATCH  # 30
N_TOKENS = GRID_H_P * GRID_W_P  # 450


def sincos_init(n_positions: int, dim: int) -> torch.Tensor:
    """Standard sin/cos positional-encoding formula, used only to INITIALIZE
    a learnable embedding table (ForeSWE's variable-embedding trick, reused
    here on the spatial-patch-index and temporal-step-index axes instead)."""
    assert dim % 2 == 0
    position = torch.arange(n_positions, dtype=torch.float64).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float64) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_positions, dim, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(torch.float32)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = N_CHANNELS, d_model: int = 128, patch: int = PATCH) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch, stride=patch)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> [B, T, N, d_model], SAME conv weights for every day
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.proj(x)  # [B*T, d_model, H/patch, W/patch]
        _, d, hp, wp = x.shape
        x = x.reshape(b, t, d, hp * wp).permute(0, 1, 3, 2)  # [B, T, N, d]
        return x


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, N, d] -- attend across T independently for each spatial token
        b, t, n, d = z.shape
        x = z.permute(0, 2, 1, 3).reshape(b * n, t, d)  # [B*N, T, d]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        x = x.reshape(b, n, t, d).permute(0, 2, 1, 3)  # back to [B, T, N, d]
        return x


class SpatialAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, N, d] -- attend across N independently for each day (plain MHA, no geography bias)
        b, t, n, d = z.shape
        x = z.reshape(b * t, n, d)  # [B*T, N, d]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        x = x.reshape(b, t, n, d)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(z)))
        return self.norm(z + self.dropout(out))


class FactorizedSTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.temporal_attn = TemporalAttention(d_model, num_heads, dropout)
        self.spatial_attn = SpatialAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.temporal_attn(z)
        z = self.spatial_attn(z)
        z = self.ffn(z)
        return z


class T0FactorizedSTTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.10,
        head_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbed(N_CHANNELS, d_model, PATCH)

        # Sin/cos-INITIALIZED, trainable positional embeddings (ForeSWE's
        # variable-embedding trick, reused here on space+time axes).
        self.spatial_pos = nn.Parameter(sincos_init(N_TOKENS, d_model).unsqueeze(0).unsqueeze(0), requires_grad=True)  # [1,1,N,d]
        self.temporal_pos = nn.Parameter(sincos_init(N_HISTORY_DAYS, d_model).unsqueeze(0).unsqueeze(2), requires_grad=True)  # [1,T,1,d]

        d_ff = d_model * ffn_ratio
        self.blocks = nn.ModuleList([FactorizedSTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T=7, C=12, H=120, W=240]
        z = self.patch_embed(x)  # [B, T, N, d]
        z = z + self.spatial_pos + self.temporal_pos
        for block in self.blocks:
            z = block(z)
        z_last = z[:, -1, :, :]  # [B, N, d] -- last timestep only (day t)
        pooled = z_last.mean(dim=1)  # [B, d]
        out = self.head(pooled).squeeze(-1)  # [B]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check (Section 16): shapes + parameter count, CPU only, no data.
    torch.manual_seed(0)
    model = T0FactorizedSTTransformer()
    n_params = count_parameters(model)
    print(f"N_TOKENS={N_TOKENS} (grid {GRID_H_P}x{GRID_W_P}), parameters={n_params}")
    x = torch.randn(2, N_HISTORY_DAYS, N_CHANNELS, GRID_H, GRID_W)
    print(f"input shape: {tuple(x.shape)}")
    z = model.patch_embed(x)
    print(f"after patch embed: {tuple(z.shape)}")
    z = z + model.spatial_pos + model.temporal_pos
    z_t = model.blocks[0].temporal_attn(z)
    print(f"after temporal attention: {tuple(z_t.shape)}")
    z_s = model.blocks[0].spatial_attn(z_t)
    print(f"after spatial attention: {tuple(z_s.shape)}")
    out = model(x)
    print(f"output shape: {tuple(out.shape)}")
