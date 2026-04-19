"""Matplotlib trajectory overlays for SDD windows (debug / qualitative review; Phase 10).

**engineering assumption:** image-plane ``x,y`` (same units as ``obs_xy`` / ``fut_xy`` / decoder means).
Primary downstream path remains CARLA CSV (:mod:`chgnet.exporters`); this module is optional tooling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from chgnet.models.chg_net import CHGNetOutput


def _t2n(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _viz_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    out = cfg.get("outputs", {})
    if not isinstance(out, dict):
        out = {}
    v = cfg.get("visualization", {})
    if not isinstance(v, dict):
        v = {}
    return {
        "dpi": int(v.get("dpi", 120)),
        "figsize": tuple(v.get("figsize_inches", (8.0, 8.0))),
        "linewidth_obs": float(v.get("linewidth_obs", 1.5)),
        "linewidth_fut": float(v.get("linewidth_fut", 1.8)),
        "linewidth_pred": float(v.get("linewidth_pred", 2.0)),
        "alpha_obs": float(v.get("alpha_obs", 0.85)),
        "alpha_fut": float(v.get("alpha_fut", 0.75)),
        "alpha_pred": float(v.get("alpha_pred", 0.9)),
        "show_legend": bool(v.get("show_legend", True)),
        "plot_gt": bool(v.get("plot_ground_truth_future", True)),
        "plot_pred": bool(v.get("plot_deterministic_prediction", True)),
        "plot_multimodal": bool(v.get("plot_multimodal_modes", False)),
        "multimodal_alpha": float(v.get("multimodal_alpha", 0.35)),
    }


def plot_batch_item_trajectories(
    batch: Mapping[str, Any],
    cfg: Mapping[str, Any],
    batch_index: int,
    model_out: CHGNetOutput | None = None,
    *,
    out_path: str | Path,
) -> None:
    """Save one PNG: trajectories for ``batch_index`` within a collated batch.

    Draws, per real agent (``agent_id >= 0``, ``class_idx >= 0``):
    - Past: ``obs_xy`` polyline where ``obs_valid``.
    - Future GT: ``fut_xy`` (dashed) where ``fut_valid`` if enabled.
    - Deterministic prediction: argmax GMM mode (solid) where ``fut_valid`` if ``model_out`` given.

    Agents with ``model_out.node_mask[b, n] == False`` are skipped when ``model_out`` is present
    (aligned with training / export masking).

    Args:
        batch: :func:`~chgnet.datasets.collate.collate_sdd_batch` dict.
        cfg: Merged config; optional ``visualization`` block (see :func:`_viz_cfg`).
        batch_index: Row in ``0 .. B-1``.
        model_out: Optional forward output for prediction overlay.
        out_path: ``.png`` path (parent dirs created).
    """
    data = cfg["data"]
    obs_len = int(data["obs_len"])
    pred_len = int(data["pred_len"])

    b = int(batch_index)
    b_max = int(batch["obs_xy"].shape[0])
    if b < 0 or b >= b_max:
        raise IndexError(f"batch_index {b} out of range for batch size {b_max}")

    obs_xy = _t2n(batch["obs_xy"][b])
    fut_xy = _t2n(batch["fut_xy"][b])
    obs_valid = _t2n(batch["obs_valid"][b]).astype(bool)
    fut_valid = _t2n(batch["fut_valid"][b]).astype(bool)
    agent_id = _t2n(batch["agent_id"][b]).astype(np.int64)
    class_idx = _t2n(batch["class_idx"][b]).astype(np.int64)

    if obs_xy.shape[0] != obs_len or fut_xy.shape[0] != pred_len:
        raise ValueError(
            f"Sample timeline shape obs_T={obs_xy.shape[0]}, fut_T={fut_xy.shape[0]} "
            f"!= cfg obs_len={obs_len}, pred_len={pred_len}."
        )

    node_mask = None
    pred_det = None
    means_mm = None
    if model_out is not None:
        node_mask = _t2n(model_out.node_mask[b]).astype(bool)
        pred_det = _t2n(model_out.gmm.deterministic_trajectory[b])
        means_mm = _t2n(model_out.gmm.means[b])

    vc = _viz_cfg(cfg)
    n_max = obs_xy.shape[1]
    scene_id = str(batch["scene_id"][b])
    seq_id = str(batch["sequence_id"][b])

    fig, ax = plt.subplots(figsize=vc["figsize"], dpi=vc["dpi"])
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])

    for n in range(n_max):
        if int(agent_id[n]) < 0 or int(class_idx[n]) < 0:
            continue
        if node_mask is not None and not bool(node_mask[n]):
            continue
        color = colors[n % len(colors)]

        oxs, oys = [], []
        for t in range(obs_len):
            if obs_valid[t, n]:
                oxs.append(float(obs_xy[t, n, 0]))
                oys.append(float(obs_xy[t, n, 1]))
        if len(oxs) >= 2:
            ax.plot(
                oxs,
                oys,
                "-",
                color=color,
                linewidth=vc["linewidth_obs"],
                alpha=vc["alpha_obs"],
            )
        elif len(oxs) == 1:
            ax.scatter(oxs, oys, c=color, s=12, alpha=vc["alpha_obs"], zorder=3)

        if vc["plot_gt"]:
            gxs, gys = [], []
            for t in range(pred_len):
                if fut_valid[t, n]:
                    gxs.append(float(fut_xy[t, n, 0]))
                    gys.append(float(fut_xy[t, n, 1]))
            if len(gxs) >= 2:
                ax.plot(
                    gxs,
                    gys,
                    "--",
                    color=color,
                    linewidth=vc["linewidth_fut"],
                    alpha=vc["alpha_fut"],
                )

        if vc["plot_pred"] and pred_det is not None:
            pxs, pys = [], []
            for t in range(pred_len):
                if fut_valid[t, n]:
                    pxs.append(float(pred_det[n, t, 0]))
                    pys.append(float(pred_det[n, t, 1]))
            if len(pxs) >= 2:
                ax.plot(
                    pxs,
                    pys,
                    "-",
                    color=color,
                    linewidth=vc["linewidth_pred"],
                    alpha=vc["alpha_pred"],
                )

        if vc["plot_multimodal"] and means_mm is not None:
            k = means_mm.shape[1]
            for k_i in range(k):
                mxs, mys = [], []
                for t in range(pred_len):
                    if fut_valid[t, n]:
                        mxs.append(float(means_mm[n, k_i, t, 0]))
                        mys.append(float(means_mm[n, k_i, t, 1]))
                if len(mxs) >= 2:
                    ax.plot(
                        mxs,
                        mys,
                        ":",
                        color=color,
                        linewidth=vc["linewidth_obs"],
                        alpha=vc["multimodal_alpha"],
                    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (image plane)")
    ax.set_ylabel("y (image plane)")
    ax.set_title(f"{scene_id} / {seq_id} (batch row {b})")
    if vc["show_legend"]:
        handles: list[Line2D] = [
            Line2D([0], [0], color="0.3", linestyle="-", linewidth=vc["linewidth_obs"], label="observed"),
        ]
        if vc["plot_gt"]:
            handles.append(
                Line2D([0], [0], color="0.3", linestyle="--", linewidth=vc["linewidth_fut"], label="GT future"),
            )
        if vc["plot_pred"] and pred_det is not None:
            handles.append(
                Line2D([0], [0], color="0.3", linestyle="-", linewidth=vc["linewidth_pred"], label="pred (det)"),
            )
        if vc["plot_multimodal"] and means_mm is not None:
            handles.append(
                Line2D([0], [0], color="0.3", linestyle=":", linewidth=1.0, label="GMM modes"),
            )
        ax.legend(handles=handles, loc="best", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
