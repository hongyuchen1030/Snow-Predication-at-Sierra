"""B0 (LSTM) and B1 (TCNN) temporal baselines for the 7-day -> 1-day Sierra
SWE-change task, referenced against Duan et al. (2024), "Using Temporal Deep
Learning Models to Estimate Daily Snow Water Equivalent Over the Rocky
Mountains" (https://github.com/ShihengDuan/code-SWE).

Duan's models operate on station-level scalar met inputs. Our daily input is
a GLOBAL atmospheric field [12,120,240], so both baselines share a lightweight
per-day spatial encoder (weights shared across the 7 history days, reusing the
downsampling conv stem pattern from delta_swe_models.py's C0/C1 baselines)
that reduces each day to a vector z_t before the temporal model runs on the
resulting length-7 sequence z_1..z_7. This isolates the LSTM-vs-TCNN temporal
comparison from the spatial representation, since both use the identical
encoder architecture.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm

N_HISTORY_DAYS = 7
N_CHANNELS = 12
ENCODER_DIM = 128


def conv_block(in_ch: int, out_ch: int, *, stride: int = 2, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.GELU(),
    )


class SharedSpatialEncoder(nn.Module):
    """X_t [12,120,240] -> z_t [D]. SAME weights applied to every history day
    (Conv2d shared across the time axis by folding T into the batch dim).
    Same downsampling depth/channels as C0_CNNControl's stem
    (delta_swe_models.py), pooled to a vector instead of kept as a feature
    map, since LSTM/TCNN (unlike ConvLSTM) consume vector sequences."""

    def __init__(self, in_channels: int = N_CHANNELS, out_dim: int = ENCODER_DIM) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            conv_block(in_channels, 32),  # 120x240 -> 60x120
            conv_block(32, 64),           # 60x120 -> 30x60
            conv_block(64, out_dim),      # 30x60 -> 15x30
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> [B, T, D]
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.stem(x)
        x = self.pool(x).flatten(1)
        return x.reshape(b, t, -1)


class B0_LSTMBaseline(nn.Module):
    """Duan-referenced LSTM: single-layer LSTM (hidden=128) -> dropout(0.3)
    -> linear regression head, applied to the shared encoder's z_1..z_7
    sequence. Uses the final hidden state (last input day)."""

    def __init__(self, hidden_size: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = SharedSpatialEncoder(out_dim=ENCODER_DIM)
        self.lstm = nn.LSTM(input_size=ENCODER_DIM, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # [B, T, D]
        _, (h_n, _) = self.lstm(z)  # h_n: [1, B, hidden]
        h_last = h_n[-1]  # last input day's hidden state
        out = self.head(self.dropout(h_last)).squeeze(-1)
        return out


class TemporalBlock(nn.Module):
    """Standard TCN residual block (Bai et al. 2018 / Duan's TCNN reference):
    two weight-normalized causal dilated Conv1d layers + ReLU + dropout, with
    a residual (1x1-conv-matched) skip connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal: pad left only, then trim
        self.pad = pad
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv1(x)[:, :, : x.shape[-1]] if self.pad > 0 else self.conv1(x)
        out = self.dropout1(self.relu(out))
        out = self.conv2(out)[:, :, : x.shape[-1]] if self.pad > 0 else self.conv2(out)
        out = self.dropout2(self.relu(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class B1_TCNNBaseline(nn.Module):
    """Duan-referenced TCNN, adapted for T=7 (see docs/... receptive-field
    note). Causal dilated Conv1d residual blocks (weight-normalized, ReLU,
    dropout) applied to the shared encoder's z_1..z_7 sequence, final
    timestep's channel vector -> scalar head.

    Duan's published config (kernel=7, levels=5, channels=64,
    dilations=2,4,8,16,32, dropout=0.4) targets Duan's much longer history
    window; blindly reusing it against T=7 would give a receptive field vastly
    exceeding the input length, wasting depth on all-zero-padded context. We
    instead use kernel=3, levels=3, dilations=1,2,4, channels=64, dropout=0.4:
    RF = 1 + 2*(kernel-1)*sum(dilations) = 1 + 2*2*(1+2+4) = 29 days, i.e. the
    smallest standard TCN depth (3 residual levels) whose receptive field
    already exceeds T=7, so every one of the 7 input days is inside the
    deepest layer's receptive field at the last timestep while keeping
    parameter count comparable to the LSTM baseline.
    """

    def __init__(self, channels: int = 64, levels: int = 3, kernel_size: int = 3, dropout: float = 0.4) -> None:
        super().__init__()
        self.encoder = SharedSpatialEncoder(out_dim=ENCODER_DIM)
        dilations = [2 ** i for i in range(levels)]  # 1,2,4
        blocks = []
        in_ch = ENCODER_DIM
        for d in dilations:
            blocks.append(TemporalBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # [B, T, D]
        z = z.transpose(1, 2)  # [B, D, T]
        out = self.tcn(z)  # [B, channels, T]
        last = out[:, :, -1]  # last (most recent) day's representation
        return self.head(last).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str) -> nn.Module:
    if name == "lstm":
        return B0_LSTMBaseline()
    if name == "tcnn":
        return B1_TCNNBaseline()
    raise ValueError(name)


if __name__ == "__main__":
    # Shape + parameter-count sanity check only (dummy random data, CPU, no
    # dataset access) -- same pattern as delta_swe_transformer_t0.py's
    # __main__ block.
    torch.manual_seed(0)
    x = torch.randn(2, N_HISTORY_DAYS, N_CHANNELS, 120, 240)
    print(f"input shape: {tuple(x.shape)}")

    for name in ("lstm", "tcnn"):
        model = build_model(name)
        n_params = count_parameters(model)
        out = model(x)
        print(f"model={name} n_params={n_params} output_shape={tuple(out.shape)}")
        if name == "tcnn":
            print(f"  tcnn kernel_size={model.kernel_size} dilations={model.dilations} receptive_field={model.receptive_field} (>= T={N_HISTORY_DAYS})")
