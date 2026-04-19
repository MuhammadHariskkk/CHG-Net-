"""Direction-aware interaction weights for CHG edges (pre–ST-GCN).

**paper-specified** (form from CHG-Net):
Let **p_i**, **p_j** be final observed positions, and use the same cos(θ_ij), cos(φ_ij) as in the edge
features (motion-direction and heading alignment).

- Distance term:  w_d(i,j) = exp(−‖**p_i** − **p_j**‖)
- Motion term:   w_v(i,j) = (1 + cos(θ_ij)) / 2
- Heading term:  w_h(i,j) = (1 + cos(φ_ij)) / 2
- Combined mask: m_ij = α·w_d + β·w_v + γ·w_h
- Masked edge:   **ẽ**_ij = m_ij · **e**_ij  (element-wise broadcast over edge feature channels)

**engineering assumption**:
- ‖**p_i** − **p_j**‖ is the Euclidean norm of **Δp** in ``edge_attr[:, :2]`` (image-plane units unless
  you rescale coordinates upstream).
- Optional hard thresholding: if ``masking.apply_threshold`` is true, set m_ij = 0 where m_ij < ``threshold``
  (validation-tuned in the paper; default off in YAML).
- Edges incident to an invalid node (``node_valid`` false) get m_ij = 0 so messages do not flow through
  padded agents (not spelled out in the paper; required for batched/padded graphs).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch

from chgnet.graph.graph_builder import CHGGraph


def _masking_params(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "apply_threshold": False,
            "threshold": 0.0,
        }
    m = cfg.get("masking", {})
    if not isinstance(m, dict):
        m = {}
    return {
        "alpha": float(m.get("alpha", 1.0)),
        "beta": float(m.get("beta", 1.0)),
        "gamma": float(m.get("gamma", 1.0)),
        "apply_threshold": bool(m.get("apply_threshold", False)),
        "threshold": float(m.get("threshold", 0.0)),
    }


def compute_direction_aware_mask(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    apply_threshold: bool = False,
    threshold: float = 0.0,
    node_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar mask m_ij per directed edge, shape ``(E,)``.

    Uses ``edge_attr[:, 0:2]`` as **Δp** = **p_j** − **p_i**, ``edge_attr[:, 4]`` as cos(θ_ij),
    ``edge_attr[:, 5]`` as cos(φ_ij) (must match :func:`chgnet.graph.features.edge_geometric_attr`).
    """
    if edge_index.shape[1] == 0:
        if edge_attr.shape[0] != 0:
            raise ValueError("edge_index has no edges but edge_attr is non-empty.")
        return edge_attr.new_zeros((0,))
    src = edge_index[0]
    dst = edge_index[1]
    e = src.numel()
    if edge_attr.shape[0] != e:
        raise ValueError(f"edge_attr rows {edge_attr.shape[0]} != num edges {e}")
    if edge_attr.shape[1] < 6:
        raise ValueError("edge_attr must have at least 6 columns (Δp, Δv, cos θ, cos φ).")

    dp = edge_attr[:, 0:2]
    dist = torch.linalg.norm(dp, dim=-1)
    w_d = torch.exp(-dist)
    cos_t = edge_attr[:, 4]
    cos_p = edge_attr[:, 5]
    w_v = 0.5 * (1.0 + cos_t)
    w_h = 0.5 * (1.0 + cos_p)

    m = (
        float(alpha) * w_d
        + float(beta) * w_v
        + float(gamma) * w_h
    )

    if node_valid is not None:
        if src.numel() > 0:
            hi = int(torch.maximum(src.max(), dst.max()).item())
            if node_valid.shape[0] <= hi:
                raise ValueError(
                    "node_valid length must be strictly greater than max node index in edge_index."
                )
        ok = node_valid[src] & node_valid[dst]
        m = m * ok.to(dtype=m.dtype)

    if apply_threshold:
        m = torch.where(m >= float(threshold), m, torch.zeros_like(m))

    return m


def compute_direction_aware_mask_from_config(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    cfg: Mapping[str, Any] | None,
    node_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same as :func:`compute_direction_aware_mask` with hyperparameters from ``cfg['masking']``."""
    p = _masking_params(cfg)
    return compute_direction_aware_mask(
        edge_index,
        edge_attr,
        alpha=p["alpha"],
        beta=p["beta"],
        gamma=p["gamma"],
        apply_threshold=p["apply_threshold"],
        threshold=p["threshold"],
        node_valid=node_valid,
    )


def mask_edge_features(edge_attr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """**ẽ** = m ⊙ **e** with ``mask`` of shape ``(E,)`` broadcast over feature columns."""
    if mask.numel() != edge_attr.shape[0]:
        raise ValueError("mask length must match number of edge rows.")
    return edge_attr * mask.unsqueeze(-1)


def apply_direction_aware_mask(
    graph: CHGGraph,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[CHGGraph, torch.Tensor]:
    """Return a new :class:`CHGGraph` with masked ``edge_attr`` and the per-edge weights ``(E,)``."""
    m = compute_direction_aware_mask_from_config(
        graph.edge_index,
        graph.edge_attr,
        cfg,
        node_valid=graph.node_valid,
    )
    masked_attr = mask_edge_features(graph.edge_attr, m)
    return replace(graph, edge_attr=masked_attr), m
