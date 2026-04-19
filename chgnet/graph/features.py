"""Node and edge geometric features for the Comprehensive Heterogeneous Graph (CHG).

**paper-specified (high level, from CHG-Net description):**
- Node cues: final observed position, velocity, class identity, heading.
- Edge cues: relative displacement, relative velocity, motion-direction cosine cos(θ_ij),
  heading-alignment cosine cos(φ_ij), plus class linkage for source/target agents.

**engineering assumption (PDF not in-repo):**
- Directed edge *i* → *j* uses source agent *i* and target *j* in the same agent index order as the
  batch (Phase 2 / collate).
- cos(θ_ij) = cosine between displacement **(p_j − p_i)** and source velocity **v_i** (interaction
  direction vs source motion). If either vector norm < ``vec_eps``, the cosine is set to **0**.
- cos(φ_ij) = dot product of heading unit vectors **u_i** = (cos h_i, sin h_i) and **u_j**, i.e.
  cos(h_i − h_j), matching heading-alignment in the paper’s notation.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


def _vec_eps(cfg: Mapping[str, Any] | None) -> float:
    g = (cfg or {}).get("graph", {}) if isinstance(cfg, dict) else {}
    if not isinstance(g, dict):
        return 1.0e-6
    return float(g.get("vec_eps", 1.0e-6))


def sanitize_agent_kinematics(
    pos: torch.Tensor,
    vel: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor | None,
    *,
    vec_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace non-finite values with 0 and expose a per-node validity mask.

    Args:
        pos: ``(N, 2)`` last-observation positions.
        vel: ``(N, 2)`` last-observation velocities.
        heading: ``(N,)`` radians.
        valid: optional ``(N,)`` bool; if None, inferred as finite(pos).all(-1).
        vec_eps: reserved for callers / cos θ denominator (see :func:`cos_theta_motion_direction`).

    Returns:
        ``pos_s, vel_s, heading_s, node_valid`` each ``(N,)`` / ``(N,2)`` as appropriate.
    """
    _ = vec_eps
    pos_s = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    vel_s = torch.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    heading_s = torch.nan_to_num(heading, nan=0.0, posinf=0.0, neginf=0.0)
    finite = (
        torch.isfinite(pos).all(dim=-1)
        & torch.isfinite(vel).all(dim=-1)
        & torch.isfinite(heading)
    )
    if valid is None:
        node_valid = finite
    else:
        node_valid = valid & finite
    return pos_s, vel_s, heading_s, node_valid


def node_geometric_tensor(
    pos: torch.Tensor,
    vel: torch.Tensor,
    heading: torch.Tensor,
) -> torch.Tensor:
    """Stack continuous node channels: ``[x, y, vx, vy, heading_rad]`` → ``(N, 5)``."""
    if pos.shape[-1] != 2 or vel.shape[-1] != 2:
        raise ValueError("pos and vel must end with dim 2.")
    if pos.shape[:-1] != vel.shape[:-1] or heading.shape != pos.shape[:-1]:
        raise ValueError("Shape mismatch between pos, vel, heading.")
    return torch.cat([pos, vel, heading.unsqueeze(-1)], dim=-1)


def cos_theta_motion_direction(
    pos: torch.Tensor,
    vel: torch.Tensor,
    edge_index: torch.Tensor,
    vec_eps: float,
) -> torch.Tensor:
    """cos(θ_ij) for each directed edge (*i* → *j*).

    Uses displacement **r** = p_j − p_i and source velocity **v_i**:
    cos θ = (r · v_i) / (‖r‖ ‖v_i‖), or **0** if either norm is below ``vec_eps``.
    """
    src = edge_index[0]
    dst = edge_index[1]
    e = src.numel()
    if e == 0:
        return pos.new_zeros((0,))
    p_i = pos[src]
    p_j = pos[dst]
    v_i = vel[src]
    r = p_j - p_i
    rn = torch.linalg.norm(r, dim=-1)
    vn = torch.linalg.norm(v_i, dim=-1)
    dot = (r * v_i).sum(dim=-1)
    ok = (rn >= vec_eps) & (vn >= vec_eps)
    out = torch.zeros(e, dtype=pos.dtype, device=pos.device)
    out[ok] = (dot[ok] / (rn[ok] * vn[ok])).clamp(-1.0, 1.0)
    return out


def cos_phi_heading_alignment(heading: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """cos(φ_ij) using heading unit vectors at source and target nodes."""
    src = edge_index[0]
    dst = edge_index[1]
    e = src.numel()
    if e == 0:
        return heading.new_zeros((0,))
    hi = heading[src]
    hj = heading[dst]
    ui = torch.stack([torch.cos(hi), torch.sin(hi)], dim=-1)
    uj = torch.stack([torch.cos(hj), torch.sin(hj)], dim=-1)
    return (ui * uj).sum(dim=-1).clamp(-1.0, 1.0)


def edge_geometric_attr(
    pos: torch.Tensor,
    vel: torch.Tensor,
    heading: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Concatenate edge channels: ``[Δp(2), Δv(2), cos_theta(1), cos_phi(1)]`` → ``(E, 6)``.

    Δp and Δv are taken from ``edge_index`` (*i* → *j*) as ``p_j - p_i`` and ``v_j - v_i`` so ordering
    matches ``cos_theta`` / ``cos_phi`` exactly.

    Class indices for endpoints are **not** included here (see :class:`chgnet.graph.graph_builder.CHGGraph`).
    """
    eps = _vec_eps(cfg)
    src = edge_index[0]
    dst = edge_index[1]
    dp = pos[dst] - pos[src]
    dv = vel[dst] - vel[src]
    ct = cos_theta_motion_direction(pos, vel, edge_index, eps)
    cp = cos_phi_heading_alignment(heading, edge_index)
    return torch.cat([dp, dv, ct.unsqueeze(-1), cp.unsqueeze(-1)], dim=-1)
