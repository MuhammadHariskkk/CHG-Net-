"""Assemble CHG tensors from Phase 2 samples or collate batches (no masking / no ST-GCN here)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from chgnet.graph.adjacency import directed_complete_edge_index
from chgnet.graph.features import (
    edge_geometric_attr,
    node_geometric_tensor,
    sanitize_agent_kinematics,
)


@dataclass
class CHGGraph:
    """One fully connected directed CHG snapshot at the last observation step.

    **Shapes**
        ``num_nodes``: ``N`` (agent slots in this graph; with padded batches, ``N`` may exceed the
        true agent count—see :func:`build_chg_graph_from_batch` ``num_nodes``). No batch dimension.
        ``node_geom``: ``(N, 5)`` — ``x, y, vx, vy, heading_rad``.
        ``node_class``: ``(N,)`` int64 class indices for embedding lookup (Phase 7).
        ``node_valid``: ``(N,)`` bool — False for padded / invalid agents (message passing should mask).
        ``agent_id``: ``(N,)`` int64 track ids from SDD (``-1`` if missing).
        ``edge_index``: ``(2, E)`` int64 with ``E = N * (N - 1)``, directed *i* → *j*, *i* ≠ *j*.
        ``edge_attr``: ``(E, 6)`` — ``Δp_xy, Δv_xy, cos_theta, cos_phi`` (see ``features.py``).
        ``edge_src_class``, ``edge_dst_class``: ``(E,)`` int64 — duplicate of node classes at endpoints
        (convenient for heterogeneous edge MLPs without gather).
    """

    num_nodes: int
    node_geom: torch.Tensor
    node_class: torch.Tensor
    node_valid: torch.Tensor
    agent_id: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_src_class: torch.Tensor
    edge_dst_class: torch.Tensor


def build_chg_graph(
    pos_last: torch.Tensor,
    vel_last: torch.Tensor,
    heading_last: torch.Tensor,
    class_idx: torch.Tensor,
    agent_id: torch.Tensor,
    node_valid: torch.Tensor | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> CHGGraph:
    """Build a CHG from last-step kinematics for ``N`` agents in fixed order.

    Args:
        pos_last: ``(N, 2)``.
        vel_last: ``(N, 2)``.
        heading_last: ``(N,)``.
        class_idx: ``(N,)`` int64 (``>= 0`` for real classes; ``-1`` padding should pair with invalid).
        agent_id: ``(N,)`` int64 SDD track ids (``-1`` if unknown/pad).
        node_valid: optional ``(N,)`` bool mask; if None, inferred from finite kinematics.
        cfg: merged config (optional ``graph.vec_eps``).
    """
    if pos_last.dim() != 2 or pos_last.shape[-1] != 2:
        raise ValueError(f"pos_last must be (N, 2), got {tuple(pos_last.shape)}")
    n = pos_last.shape[0]
    if n == 0:
        device = pos_last.device
        z = torch.zeros(0, dtype=torch.long, device=device)
        empty = torch.zeros((0, 5), dtype=pos_last.dtype, device=device)
        return CHGGraph(
            num_nodes=0,
            node_geom=empty,
            node_class=z,
            node_valid=z.bool(),
            agent_id=z,
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=device),
            edge_attr=torch.zeros((0, 6), dtype=pos_last.dtype, device=device),
            edge_src_class=z,
            edge_dst_class=z,
        )

    if isinstance(cfg, dict):
        gcfg = (cfg or {}).get("graph", {})
        if not isinstance(gcfg, dict):
            gcfg = {}
    else:
        gcfg = {}
    vec_eps = float(gcfg.get("vec_eps", 1.0e-6))
    pos_s, vel_s, head_s, finite_valid = sanitize_agent_kinematics(
        pos_last, vel_last, heading_last, node_valid, vec_eps=vec_eps
    )
    cls_ok = class_idx >= 0
    aid_ok = agent_id >= 0
    valid = finite_valid & cls_ok & aid_ok

    node_geom = node_geometric_tensor(pos_s, vel_s, head_s)
    edge_index = directed_complete_edge_index(n, device=pos_last.device)
    edge_attr = edge_geometric_attr(pos_s, vel_s, head_s, edge_index, cfg)

    src = edge_index[0]
    dst = edge_index[1]
    edge_src_class = class_idx[src]
    edge_dst_class = class_idx[dst]

    return CHGGraph(
        num_nodes=n,
        node_geom=node_geom,
        node_class=class_idx,
        node_valid=valid,
        agent_id=agent_id,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_src_class=edge_src_class,
        edge_dst_class=edge_dst_class,
    )


def build_chg_graph_from_batch(
    batch: Mapping[str, Any],
    batch_idx: int,
    last_obs_index: int = -1,
    cfg: Mapping[str, Any] | None = None,
    num_nodes: int | None = None,
) -> CHGGraph:
    """Build CHG from :func:`chgnet.datasets.collate.collate_sdd_batch` output.

    Args:
        num_nodes: If set (e.g. collated ``N_max``), build a graph on that many slots so every batch
            row shares the same ``edge_index`` topology for batched ST-GCN. Padded slots should
            carry invalid kinematics / ``class_idx == -1`` / ``obs_valid`` false (Phase 7).
            If ``None`` (default), uses ``num_agents[batch_idx]`` — backward-compatible with
            per-sample exact agent counts.
    """
    b = int(batch_idx)
    cap = int(batch["obs_xy"].shape[2])
    if num_nodes is None:
        n = int(batch["num_agents"][b].item())
    else:
        n = int(num_nodes)
        if n > cap:
            n = cap
    if n <= 0:
        raise ValueError(f"graph node count must be positive, got n={n} (batch_idx={b}).")

    pos = batch["obs_xy"][b, last_obs_index, :n, :].float()
    vel = batch["obs_vel"][b, last_obs_index, :n, :].float()
    head = batch["obs_heading"][b, last_obs_index, :n].float()
    cls = batch["class_idx"][b, :n].long()
    aid = batch["agent_id"][b, :n].long()
    vmask = batch["obs_valid"][b, last_obs_index, :n]

    return build_chg_graph(pos, vel, head, cls, aid, node_valid=vmask.bool(), cfg=cfg)


def build_chg_graph_from_sample_dict(
    sample: Mapping[str, Any],
    last_obs_index: int = -1,
    cfg: Mapping[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> CHGGraph:
    """Build CHG from a single Phase 2 sample dict (NumPy arrays inside)."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    obs_xy = torch.as_tensor(sample["obs_xy"], dtype=torch.float32, device=dev)
    obs_vel = torch.as_tensor(sample["obs_vel"], dtype=torch.float32, device=dev)
    obs_head = torch.as_tensor(sample["obs_heading"], dtype=torch.float32, device=dev)
    cls = torch.as_tensor(sample["class_idx"], dtype=torch.long, device=dev)
    aid = torch.as_tensor(sample["agent_id"], dtype=torch.long, device=dev)
    valid = torch.as_tensor(sample["obs_valid"], dtype=torch.bool, device=dev)

    n = int(obs_xy.shape[1])
    pos = obs_xy[last_obs_index, :n]
    vel = obs_vel[last_obs_index, :n]
    head = obs_head[last_obs_index, :n]
    vmask = valid[last_obs_index, :n]

    return build_chg_graph(pos, vel, head, cls, aid, node_valid=vmask, cfg=cfg)
