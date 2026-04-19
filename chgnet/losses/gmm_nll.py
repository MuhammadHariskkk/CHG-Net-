"""GMM trajectory negative log-likelihood for :class:`~chgnet.models.gmm_decoder.GMMDecoderOutput`.

**paper-specified (high level):** train the mixture-of-Gaussians head with a likelihood-based objective.

**engineering assumption:**
- One mixture weight vector per agent for the **entire** future horizon (matches ``mix_logits`` shape ``(B, N, K)``).
- Given mode *k*, future steps are **conditionally independent** diagonal bivariate Gaussians:
  log p(y | k) = sum over time and (x, y) of univariate log-densities.
- Targets are **absolute** image-plane positions aligned with preprocessed ``fut_xy`` (same frame as decoder means).
- An agent contributes to the loss only if ``node_mask`` is true and **every** future step is ``fut_valid``
  (matches ``data.require_full_window_valid`` semantics; padded / invalid agents are excluded).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as Fnn

from chgnet.models.gmm_decoder import GMMDecoderOutput


def gmm_trajectory_nll(
    gmm: GMMDecoderOutput,
    fut_xy: torch.Tensor,
    *,
    node_mask: torch.Tensor,
    fut_valid: torch.Tensor,
) -> torch.Tensor:
    """Scalar mean NLL over agents with fully valid futures and ``node_mask`` true.

    Args:
        gmm: Decoder output; ``means`` / ``std`` are ``(B, N, K, L, 2)``.
        fut_xy: ``(B, pred_len, N, 2)`` — collate layout.
        node_mask: ``(B, N)`` bool (e.g. :attr:`~chgnet.models.chg_net.CHGNetOutput.node_mask`).
        fut_valid: ``(B, pred_len, N)`` bool.

    Returns:
        Scalar tensor (differentiable): mean over selected agents of
        ``-log sum_k pi_k * exp(log p(y | k))``.
    """
    if fut_xy.dim() != 4 or fut_xy.shape[-1] != 2:
        raise ValueError(f"fut_xy must be (B, L, N, 2), got {tuple(fut_xy.shape)}")
    b, l, n, _ = fut_xy.shape
    y = fut_xy.permute(0, 2, 1, 3).contiguous()

    means = gmm.means
    std = gmm.std
    logits = gmm.mix_logits
    if means.shape[0] != b or means.shape[1] != n:
        raise ValueError(
            f"gmm batch/agent dims {means.shape[:2]} != fut {(b, n)}"
        )
    k = means.shape[2]
    if logits.shape != (b, n, k) or means.shape[3] != l or means.shape[4] != 2:
        raise ValueError(
            f"gmm shape mismatch: means {tuple(means.shape)}, logits {tuple(logits.shape)}, L={l}"
        )

    y_exp = y.unsqueeze(2).expand(b, n, k, l, 2)
    var = std * std
    log_2pi_var = torch.log(2.0 * math.pi * var)
    quad = ((y_exp - means) ** 2) / var
    log_p_step = -0.5 * (quad + log_2pi_var).sum(dim=-1)
    log_p_traj = log_p_step.sum(dim=-1)

    log_pi = Fnn.log_softmax(logits, dim=-1)
    log_mix = torch.logsumexp(log_pi + log_p_traj, dim=-1)
    nll = -log_mix

    step_ok = fut_valid.permute(0, 2, 1)
    agent_ok = node_mask & step_ok.all(dim=-1)
    w = agent_ok.to(dtype=nll.dtype)
    denom = w.sum().clamp_min(1.0)
    # Padded agents can still carry a forward NLL tensor slot; 0 * nan would poison the sum.
    nll_safe = torch.nan_to_num(nll, nan=0.0, posinf=0.0, neginf=0.0)
    return (nll_safe * w).sum() / denom
