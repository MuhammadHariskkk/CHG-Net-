"""Micro-TCN: dilated 1D convolutions with residuals over the observation timeline.

**paper-specified:** kernel size 3, two layers, dilated temporal Conv1d with residual refinement
(high-level CHG-Net description).

**engineering assumption:**
- Layout is ``(B, C, T)`` (PyTorch Conv1d convention) with ``T = data.obs_len`` (e.g. 8).
- Dilation per layer *ℓ* is ``dilation_base**ℓ`` (default ``dilation_base=2`` → 1, then 2).
- Same-length convolutions: ``padding = dilation * (kernel_size // 2)`` for odd ``kernel_size``.
- Residual: ``output = ReLU(x + conv(ReLU(conv(x))))`` is avoided; we use **pre-activation style**
  ``ReLU(x + conv(x))`` per dilated block (two blocks in sequence). If ``in_channels != out_channels``,
  a 1×1 Conv1d projects the shortcut (not needed when both are ``graph_hidden_dim``).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedResidualBlock1d(nn.Module):
    """One dilated Conv1d with ReLU and additive residual (same channels)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for symmetric padding.")
        pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv(x), inplace=True)


class MicroTCN(nn.Module):
    """Stack of dilated residual Conv1d blocks (default: 2 layers, kernel 3)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 3,
        num_layers: int = 2,
        dilation_base: int = 2,
    ) -> None:
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        if in_channels != out_channels:
            self.in_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.in_proj = nn.Identity()

        blocks: list[nn.Module] = []
        for layer in range(num_layers):
            dil = int(dilation_base**layer)
            blocks.append(_DilatedResidualBlock1d(out_channels, kernel_size, dil))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``(B, C_in, T)`` observation sequence per agent (or flattened batch of agents).

        Returns:
            ``(B, C_out, T)`` with ``T`` unchanged.
        """
        if x.dim() != 3:
            raise ValueError(f"MicroTCN expects (B, C, T), got {tuple(x.shape)}")
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


def micro_tcn_from_config(cfg: Mapping[str, Any] | None, in_channels: int | None = None) -> MicroTCN:
    """Build :class:`MicroTCN` from merged YAML config (``model.micro_tcn``, ``graph_hidden_dim``)."""
    if cfg is None:
        c = 64
        return MicroTCN(c, c, kernel_size=3, num_layers=2, dilation_base=2)
    m = cfg.get("model", {})
    if not isinstance(m, dict):
        m = {}
    mt = m.get("micro_tcn", {})
    if not isinstance(mt, dict):
        mt = {}
    c = int(in_channels if in_channels is not None else m.get("graph_hidden_dim", 64))
    return MicroTCN(
        in_channels=c,
        out_channels=c,
        kernel_size=int(mt.get("kernel_size", 3)),
        num_layers=int(mt.get("num_layers", 2)),
        dilation_base=int(mt.get("dilation_base", 2)),
    )
