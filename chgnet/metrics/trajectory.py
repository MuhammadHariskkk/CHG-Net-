"""ADE / FDE on deterministic (argmax-mixture) predictions (**paper-specified** inference default)."""

from __future__ import annotations

import torch


def ade_fde_deterministic(
    pred_xy: torch.Tensor,
    fut_xy: torch.Tensor,
    *,
    node_mask: torch.Tensor,
    fut_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average / final displacement errors in the same units as ``fut_xy``.

    Args:
        pred_xy: ``(B, N, pred_len, 2)`` — e.g. :attr:`GMMDecoderOutput.deterministic_trajectory`.
        fut_xy: ``(B, pred_len, N, 2)`` — collate layout.
        node_mask: ``(B, N)`` bool.
        fut_valid: ``(B, pred_len, N)`` bool.

    Returns:
        ``(ade, fde)`` scalars, both averaged only over agents with ``node_mask`` and **all**
        ``fut_valid`` steps true — same agent set as :func:`chgnet.losses.gmm_nll.gmm_trajectory_nll`.
        ADE is the mean over those agents of the per-agent mean L2 over time; FDE is the L2 at the
        last step on that same set.
    """
    if pred_xy.dim() != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(f"pred_xy must be (B, N, L, 2), got {tuple(pred_xy.shape)}")
    if fut_xy.dim() != 4 or fut_xy.shape[-1] != 2:
        raise ValueError(f"fut_xy must be (B, L, N, 2), got {tuple(fut_xy.shape)}")
    b, n, l, _ = pred_xy.shape
    if fut_xy.shape[0] != b or fut_xy.shape[2] != n or fut_xy.shape[1] != l:
        raise ValueError(
            f"shape mismatch pred {tuple(pred_xy.shape)} vs fut {tuple(fut_xy.shape)}"
        )

    tgt = fut_xy.permute(0, 2, 1, 3).contiguous()
    dist = torch.linalg.norm(pred_xy - tgt, dim=-1)
    # Collate pads ``fut_xy`` with NaN; invalid slots must not poison means or 0-weighted sums.
    dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    step_ok = fut_valid.permute(0, 2, 1)
    agent_ok = node_mask & step_ok.all(dim=-1)
    w_agent = agent_ok.to(dtype=dist.dtype)
    denom_a = w_agent.sum().clamp_min(1.0)
    per_agent_mean_dist = dist.mean(dim=-1)
    ade = (per_agent_mean_dist * w_agent).sum() / denom_a
    fde = (dist[:, :, -1] * w_agent).sum() / denom_a
    return ade, fde
