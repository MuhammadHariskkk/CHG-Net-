"""GMM trajectory decoder: K bivariate Gaussian modes per agent (diagonal covariances).

**paper-specified:** mixture of *K* Gaussian components (default *K* = 3); deterministic prediction
uses the component with **largest mixture weight** (argmax over softmax logits).

**engineering assumption:**
- MLP width: if ``hidden_dim`` is passed to :class:`GMMTrajectoryDecoder`, it is used; otherwise
  ``max(64, 4 * input_dim)``. :func:`gmm_decoder_from_config` sets
  ``hidden_dim = max(64, 4 * input_dim, graph_hidden_dim)`` from YAML.
- Standard deviations ``σ = softplus(raw) + min_std`` with ``min_std = 1e-4`` for stability.
- Means are **raw linear outputs** in the same units as preprocessed future positions ``fut_xy``
  (image-plane absolutes; see :func:`chgnet.losses.gmm_nll.gmm_trajectory_nll`).

**Output shapes** (always with an agent dimension ``N``; use ``N=1`` for single-agent batches):
    ``mix_logits``: ``(B, N, K)``
    ``means``: ``(B, N, K, pred_len, 2)``
    ``std``: ``(B, N, K, pred_len, 2)`` — diagonal σ per coordinate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as Fnn


@dataclass
class GMMDecoderOutput:
    """Bundle of tensors for loss, metrics, and CARLA export (Phase 9)."""

    mix_logits: torch.Tensor
    means: torch.Tensor
    std: torch.Tensor

    @property
    def mix_probs(self) -> torch.Tensor:
        """``(B, N, K)`` mixture probabilities (softmax over ``K``)."""
        return Fnn.softmax(self.mix_logits, dim=-1)

    @property
    def deterministic_mode_idx(self) -> torch.Tensor:
        """``(B, N)`` long — argmax over mixture (``paper-specified`` default inference)."""
        return torch.argmax(self.mix_probs, dim=-1)

    @property
    def deterministic_trajectory(self) -> torch.Tensor:
        """``(B, N, pred_len, 2)`` means at :attr:`deterministic_mode_idx`."""
        return select_mode_trajectory(self.means, self.deterministic_mode_idx)

    @property
    def all_mode_trajectories(self) -> torch.Tensor:
        """Alias for ``means`` (explicit name for exporters)."""
        return self.means


def select_mode_trajectory(
    means: torch.Tensor,
    mode_idx: torch.Tensor,
) -> torch.Tensor:
    """Gather ``means[b,n,mode_idx[b,n], :, :]`` → ``(B, N, L, 2)``."""
    if means.dim() != 5:
        raise ValueError(f"means must be (B, N, K, L, 2), got {tuple(means.shape)}")
    b, n, k, l, _ = means.shape
    if mode_idx.shape != (b, n):
        raise ValueError(f"mode_idx must be (B, N)={(b, n)}, got {tuple(mode_idx.shape)}")
    bn = b * n
    mf = means.reshape(bn, k, l, 2)
    mid = mode_idx.reshape(bn)
    idx = mid.view(bn, 1, 1, 1).expand(bn, 1, l, 2).long()
    out = torch.gather(mf, dim=1, index=idx).squeeze(1)
    return out.reshape(b, n, l, 2)


class GMMTrajectoryDecoder(nn.Module):
    """Predict K future trajectories (per step bivariate diagonal Gaussian)."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int = 12,
        num_modes: int = 3,
        coord_dim: int = 2,
        hidden_dim: int | None = None,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = int(pred_len)
        self.num_modes = int(num_modes)
        self.coord_dim = int(coord_dim)
        self.min_std = float(min_std)
        h = hidden_dim if hidden_dim is not None else max(64, 4 * input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h),
            nn.ReLU(inplace=True),
        )
        flat_mean = num_modes * pred_len * coord_dim
        flat_std = flat_mean
        self.head_logits = nn.Linear(h, num_modes)
        self.head_mean = nn.Linear(h, flat_mean)
        self.head_std_raw = nn.Linear(h, flat_std)

    def forward(self, h: torch.Tensor) -> GMMDecoderOutput:
        """Args:
            h: ``(B, N, D)`` context per agent, or ``(B, D)`` (treated as ``N=1``).

        Returns:
            :class:`GMMDecoderOutput` with **N** preserved (``N=1`` if input was 2D).
        """
        if h.dim() == 2:
            h = h.unsqueeze(1)
        elif h.dim() != 3:
            raise ValueError(f"h must be (B, D) or (B, N, D), got {tuple(h.shape)}")

        b, n, d = h.shape
        if d != self.input_dim:
            raise ValueError(f"last dim {d} != input_dim {self.input_dim}")

        z = self.encoder(h)
        logits = self.head_logits(z)
        mean_flat = self.head_mean(z)
        std_raw = self.head_std_raw(z)

        k, l, c = self.num_modes, self.pred_len, self.coord_dim
        means = mean_flat.view(b, n, k, l, c)
        std = Fnn.softplus(std_raw.view(b, n, k, l, c)) + self.min_std

        return GMMDecoderOutput(mix_logits=logits, means=means, std=std)


def gmm_decoder_from_config(
    cfg: Mapping[str, Any] | None,
    input_dim: int,
    *,
    pred_len: int | None = None,
) -> GMMTrajectoryDecoder:
    """Build decoder from config: ``model.num_gmm_components``, ``data.pred_len``, hidden width."""
    k = 3
    pl = 12
    gh = 64
    if cfg is not None:
        m = cfg.get("model", {})
        d = cfg.get("data", {})
        if isinstance(m, dict):
            k = int(m.get("num_gmm_components", k))
            gh = int(m.get("graph_hidden_dim", gh))
        if isinstance(d, dict) and pred_len is None:
            pl = int(d.get("pred_len", pl))
    if pred_len is not None:
        pl = int(pred_len)
    hidden = max(64, 4 * input_dim, gh)
    return GMMTrajectoryDecoder(
        input_dim=input_dim,
        pred_len=pl,
        num_modes=k,
        hidden_dim=hidden,
    )
