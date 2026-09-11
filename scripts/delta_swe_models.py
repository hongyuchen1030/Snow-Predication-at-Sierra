"""C0 (ordinary CNN control) and C1 (ConvLSTM spatiotemporal baseline) for
the 7-day -> 1-day Sierra SWE-change task. Both take [B,7,12,120,240] and
output a scalar (normalized) ΔSWE prediction, shape [B].
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int, *, stride: int = 2, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.GELU(),
    )


class C0_CNNControl(nn.Module):
    """Ordinary 2D CNN: time folded into channels [B,7*12=84,120,240]. No attention, no recurrence."""

    def __init__(self, n_history_days: int = 7, n_channels: int = 12) -> None:
        super().__init__()
        in_ch = n_history_days * n_channels  # 84
        self.stem = conv_block(in_ch, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 128, stride=2)  # optional extra lightweight block
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b, t * c, h, w)
        x = self.stem(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).flatten(1)
        return self.mlp(x).squeeze(-1)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.hidden_ch, height, width)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


class C1_ConvLSTMBaseline(nn.Module):
    """Shared per-day spatial stem (12,120,240) -> (C',30,60), then a ConvLSTM
    over the 7-day sequence at reduced resolution, then pooling + MLP head."""

    def __init__(self, n_history_days: int = 7, n_channels: int = 12, stem_ch: int = 64, hidden_ch: int = 64, n_convlstm_layers: int = 1) -> None:
        super().__init__()
        self.n_history_days = n_history_days
        self.stem = nn.Sequential(
            conv_block(n_channels, 32),      # 120x240 -> 60x120
            conv_block(32, stem_ch),          # 60x120 -> 30x60
        )
        self.convlstm_layers = nn.ModuleList(
            [ConvLSTMCell(stem_ch if i == 0 else hidden_ch, hidden_ch) for i in range(n_convlstm_layers)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_ch, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        feat = self.stem(x_flat)  # [b*t, stem_ch, 30, 60]
        _, cprime, hh, ww = feat.shape
        feat = feat.reshape(b, t, cprime, hh, ww)

        states = [layer.init_state(b, hh, ww, x.device) for layer in self.convlstm_layers]
        for step in range(t):
            inp = feat[:, step]
            for li, layer in enumerate(self.convlstm_layers):
                h_next, c_next = layer(inp, states[li])
                states[li] = (h_next, c_next)
                inp = h_next
        final_h = states[-1][0]  # [b, hidden_ch, 30, 60]
        pooled = self.pool(final_h).flatten(1)
        return self.mlp(pooled).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str) -> nn.Module:
    if name == "cnn":
        return C0_CNNControl()
    if name == "convlstm":
        return C1_ConvLSTMBaseline()
    raise ValueError(name)
