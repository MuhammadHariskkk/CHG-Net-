"""Build CARLA-oriented CSV rows from collate batches and optional :class:`~chgnet.models.chg_net.CHGNetOutput`."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from chgnet.exporters.schema import (
    PHASE_GROUND_TRUTH,
    PHASE_OBSERVED,
    PHASE_PREDICTED,
    PHASE_SORT_ORDER,
    SOURCE_DATASET,
    SOURCE_MODEL,
    default_column_order,
    normalize_export_modes,
    validate_row_minimal,
)
from chgnet.models.chg_net import CHGNetOutput


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_scene_filename(scene_id: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", scene_id, flags=re.UNICODE)[:200] or "scene"


def load_world_affine(path: str | Path) -> Callable[[float, float], tuple[float, float]]:
    """Load ``offset_*`` / ``scale_*`` from YAML/JSON for image-plane → world stub.

    **engineering assumption:** affine map only; full homography can be added later.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"world_transform_path not found: {path.resolve()}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"Transform file root must be a mapping: {path}")
    ox = float(data.get("offset_x", 0.0))
    oy = float(data.get("offset_y", 0.0))
    sx = float(data.get("scale_x", 1.0))
    sy = float(data.get("scale_y", 1.0))

    def apply(x: float, y: float) -> tuple[float, float]:
        return (sx * x + ox, sy * y + oy)

    return apply


def _world_coords(
    x: float,
    y: float,
    *,
    coordinate_mode: str,
    transform: Callable[[float, float], tuple[float, float]] | None,
) -> tuple[float, float]:
    if coordinate_mode == "raw":
        return (x, y)
    if coordinate_mode == "world":
        if transform is None:
            raise ValueError("coordinate_mode 'world' requires a loaded world transform.")
        return transform(x, y)
    raise ValueError(f"Unknown coordinate_mode: {coordinate_mode!r}")


def _base_row(
    *,
    scene_id: str,
    sequence_id: str,
    agent_id: int,
    agent_class: str,
    raw_label: str,
    mapped_label: str,
    frame_index: int,
    time_index: int,
    sample_rate_hz: float,
    phase: str,
    mode_id: int,
    mode_probability: float,
    x: float,
    y: float,
    vx: float,
    vy: float,
    heading_rad: float,
    is_prediction: bool,
    is_deterministic: bool,
    source: str,
    coordinate_mode: str,
    transform: Callable[[float, float], tuple[float, float]] | None,
    export_run_id: str,
    track_valid: bool,
    confidence: float,
) -> dict[str, Any]:
    ts = float(time_index) / float(sample_rate_hz) if sample_rate_hz > 0 else 0.0
    wx, wy = _world_coords(x, y, coordinate_mode=coordinate_mode, transform=transform)
    row: dict[str, Any] = {
        "scene_id": scene_id,
        "sequence_id": sequence_id,
        "agent_id": int(agent_id),
        "agent_class": agent_class,
        "raw_label": raw_label,
        "mapped_label": mapped_label,
        "frame_index": int(frame_index),
        "time_index": int(time_index),
        "timestamp_sec": ts,
        "phase": phase,
        "mode_id": int(mode_id),
        "mode_probability": float(mode_probability),
        "x": float(x),
        "y": float(y),
        "vx": float(vx),
        "vy": float(vy),
        "heading_rad": float(heading_rad),
        "is_prediction": bool(is_prediction),
        "is_deterministic": bool(is_deterministic),
        "source": source,
        "scene_name": scene_id,
        "sample_rate_hz": float(sample_rate_hz),
        "world_x": float(wx),
        "world_y": float(wy),
        "confidence": float(confidence),
        "track_valid": bool(track_valid),
        "export_run_id": export_run_id,
    }
    validate_row_minimal(row)
    return row


def _agent_is_real(agent_id: int, class_idx: int) -> bool:
    return int(agent_id) >= 0 and int(class_idx) >= 0


def _finite_or_zero(v: float) -> float:
    return float(v) if math.isfinite(v) else 0.0


def build_export_rows(
    batch: Mapping[str, Any],
    model_out: CHGNetOutput | None,
    cfg: Mapping[str, Any],
    *,
    export_run_id: str = "",
) -> list[dict[str, Any]]:
    """Flatten one collated batch to CARLA CSV row dicts.

    Uses ``carla_export.*`` and ``data.obs_len`` / ``data.pred_len`` from ``cfg``. Respects
    ``include_observed``, ``include_ground_truth``, ``include_prediction`` and ``export_modes``
    (see :func:`chgnet.exporters.schema.normalize_export_modes`).

    **Contract:** ``agent_class`` and ``mapped_label`` are identical strings on every row.

    Args:
        batch: Output of :func:`chgnet.datasets.collate.collate_sdd_batch` (CPU tensors OK).
        model_out: Forward output if predictions are exported; required when prediction rows are emitted.
        cfg: Merged config.
        export_run_id: Optional run id column for reproducibility.
    """
    data = cfg["data"]
    obs_len = int(data["obs_len"])
    pred_len = int(data["pred_len"])
    hz_data = float(data.get("sample_rate_hz", 2.5))

    ce = cfg.get("carla_export", {})
    if not isinstance(ce, dict):
        ce = {}
    hz = float(ce.get("sample_rate_hz", hz_data))
    coordinate_mode = str(ce.get("coordinate_mode", "raw"))
    wpath = ce.get("world_transform_path")
    transform: Callable[[float, float], tuple[float, float]] | None = None
    if wpath:
        transform = load_world_affine(Path(wpath))
    elif coordinate_mode == "world":
        raise ValueError("carla_export.coordinate_mode is 'world' but world_transform_path is null.")

    include_obs_flag = bool(ce.get("include_observed", True))
    include_gt_flag = bool(ce.get("include_ground_truth", False))
    include_pred = bool(ce.get("include_prediction", True))
    modes = normalize_export_modes(ce.get("export_modes"))
    has_combined = "combined" in modes
    has_ground_truth_mode = "ground_truth" in modes
    has_deterministic = "deterministic" in modes
    has_multimodal = "multimodal" in modes

    include_obs = include_obs_flag or has_combined
    include_gt = include_gt_flag or has_ground_truth_mode or has_combined
    emit_det = include_pred and (has_deterministic or has_combined)
    emit_mm = include_pred and has_multimodal

    obs_xy = _to_numpy(batch["obs_xy"])
    obs_vel = _to_numpy(batch["obs_vel"])
    obs_heading = _to_numpy(batch["obs_heading"])
    obs_valid = _to_numpy(batch["obs_valid"])
    fut_xy = _to_numpy(batch["fut_xy"])
    fut_valid = _to_numpy(batch["fut_valid"])
    slot_np = _to_numpy(batch["slot_indices"])
    agent_id_np = _to_numpy(batch["agent_id"])
    class_idx_np = _to_numpy(batch["class_idx"])

    scene_ids: list[str] = list(batch["scene_id"])
    seq_ids: list[str] = list(batch["sequence_id"])
    raw_lists: list[list[str]] = batch["raw_label"]
    map_lists: list[list[str]] = batch["mapped_label"]

    B = obs_xy.shape[0]
    n_max = obs_xy.shape[2]
    if obs_xy.shape[1] != obs_len or fut_xy.shape[1] != pred_len:
        raise ValueError(
            f"Batch T dims (obs_T={obs_xy.shape[1]}, fut_T={fut_xy.shape[1]}) != "
            f"cfg data.obs_len={obs_len}, data.pred_len={pred_len}."
        )
    need_slots = obs_len + pred_len
    if slot_np.shape[0] != B or slot_np.shape[1] != need_slots:
        raise ValueError(
            f"slot_indices shape {slot_np.shape} != (B={B}, obs_len+pred_len={need_slots})."
        )

    mix_probs_np: np.ndarray | None = None
    means_np: np.ndarray | None = None
    det_idx_np: np.ndarray | None = None
    if model_out is not None:
        mix_probs_np = _to_numpy(model_out.gmm.mix_probs)
        means_np = _to_numpy(model_out.gmm.means)
        det_idx_np = _to_numpy(model_out.gmm.deterministic_mode_idx)
    node_mask_np: np.ndarray | None = None
    if model_out is not None:
        node_mask_np = _to_numpy(model_out.node_mask)

    rows: list[dict[str, Any]] = []

    if emit_det or emit_mm:
        if model_out is None:
            raise ValueError("Prediction export requires model_out (forward pass).")

    def emit_obs(b: int, n: int) -> None:
        if not include_obs:
            return
        sid, sq = scene_ids[b], seq_ids[b]
        raw_s, map_s = raw_lists[b][n], map_lists[b][n]
        agent_class = map_s
        slots = slot_np[b]
        for t in range(obs_len):
            if not bool(obs_valid[b, t, n]):
                continue
            k = int(slots[t])
            x, y = float(obs_xy[b, t, n, 0]), float(obs_xy[b, t, n, 1])
            vx = _finite_or_zero(float(obs_vel[b, t, n, 0]))
            vy = _finite_or_zero(float(obs_vel[b, t, n, 1]))
            hd = _finite_or_zero(float(obs_heading[b, t, n]))
            rows.append(
                _base_row(
                    scene_id=sid,
                    sequence_id=sq,
                    agent_id=int(agent_id_np[b, n]),
                    agent_class=agent_class,
                    raw_label=raw_s,
                    mapped_label=map_s,
                    frame_index=k,
                    time_index=k,
                    sample_rate_hz=hz,
                    phase=PHASE_OBSERVED,
                    mode_id=0,
                    mode_probability=1.0,
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    heading_rad=hd,
                    is_prediction=False,
                    is_deterministic=True,
                    source=SOURCE_DATASET,
                    coordinate_mode=coordinate_mode,
                    transform=transform,
                    export_run_id=export_run_id,
                    track_valid=True,
                    confidence=1.0,
                )
            )

    def emit_gt(b: int, n: int) -> None:
        if not include_gt:
            return
        if node_mask_np is not None and not bool(node_mask_np[b, n]):
            return
        sid, sq = scene_ids[b], seq_ids[b]
        raw_s, map_s = raw_lists[b][n], map_lists[b][n]
        agent_class = map_s
        slots = slot_np[b]
        last_xy = obs_xy[b, obs_len - 1, n].astype(np.float64)
        for t in range(pred_len):
            if not bool(fut_valid[b, t, n]):
                continue
            k = int(slots[obs_len + t])
            x, y = float(fut_xy[b, t, n, 0]), float(fut_xy[b, t, n, 1])
            if t == 0:
                vx = (x - float(last_xy[0])) * hz
                vy = (y - float(last_xy[1])) * hz
            else:
                px, py = float(fut_xy[b, t - 1, n, 0]), float(fut_xy[b, t - 1, n, 1])
                vx = (x - px) * hz
                vy = (y - py) * hz
            hd = math.atan2(vy, vx + 1e-8) if (vx != 0 or vy != 0) else 0.0
            rows.append(
                _base_row(
                    scene_id=sid,
                    sequence_id=sq,
                    agent_id=int(agent_id_np[b, n]),
                    agent_class=agent_class,
                    raw_label=raw_s,
                    mapped_label=map_s,
                    frame_index=k,
                    time_index=k,
                    sample_rate_hz=hz,
                    phase=PHASE_GROUND_TRUTH,
                    mode_id=0,
                    mode_probability=1.0,
                    x=x,
                    y=y,
                    vx=float(vx),
                    vy=float(vy),
                    heading_rad=float(hd),
                    is_prediction=False,
                    is_deterministic=True,
                    source=SOURCE_DATASET,
                    coordinate_mode=coordinate_mode,
                    transform=transform,
                    export_run_id=export_run_id,
                    track_valid=True,
                    confidence=1.0,
                )
            )

    def emit_pred_det(b: int, n: int) -> None:
        if not emit_det:
            return
        assert mix_probs_np is not None and means_np is not None and det_idx_np is not None
        if node_mask_np is not None and not bool(node_mask_np[b, n]):
            return
        sid, sq = scene_ids[b], seq_ids[b]
        raw_s, map_s = raw_lists[b][n], map_lists[b][n]
        agent_class = map_s
        slots = slot_np[b]
        dk = int(det_idx_np[b, n])
        for t in range(pred_len):
            if not bool(fut_valid[b, t, n]):
                continue
            k = int(slots[obs_len + t])
            x = float(means_np[b, n, dk, t, 0])
            y = float(means_np[b, n, dk, t, 1])
            pi = float(mix_probs_np[b, n, dk])
            if t == 0:
                lx, ly = float(obs_xy[b, obs_len - 1, n, 0]), float(obs_xy[b, obs_len - 1, n, 1])
                vx = (x - lx) * hz
                vy = (y - ly) * hz
            else:
                px = float(means_np[b, n, dk, t - 1, 0])
                py = float(means_np[b, n, dk, t - 1, 1])
                vx = (x - px) * hz
                vy = (y - py) * hz
            hd = math.atan2(vy, vx + 1e-8) if (vx != 0 or vy != 0) else 0.0
            rows.append(
                _base_row(
                    scene_id=sid,
                    sequence_id=sq,
                    agent_id=int(agent_id_np[b, n]),
                    agent_class=agent_class,
                    raw_label=raw_s,
                    mapped_label=map_s,
                    frame_index=k,
                    time_index=k,
                    sample_rate_hz=hz,
                    phase=PHASE_PREDICTED,
                    mode_id=dk,
                    mode_probability=pi,
                    x=x,
                    y=y,
                    vx=float(vx),
                    vy=float(vy),
                    heading_rad=float(hd),
                    is_prediction=True,
                    is_deterministic=True,
                    source=SOURCE_MODEL,
                    coordinate_mode=coordinate_mode,
                    transform=transform,
                    export_run_id=export_run_id,
                    track_valid=bool(fut_valid[b, t, n]),
                    confidence=pi,
                )
            )

    def emit_pred_mm(b: int, n: int) -> None:
        if not emit_mm:
            return
        assert mix_probs_np is not None and means_np is not None
        if node_mask_np is not None and not bool(node_mask_np[b, n]):
            return
        sid, sq = scene_ids[b], seq_ids[b]
        raw_s, map_s = raw_lists[b][n], map_lists[b][n]
        agent_class = map_s
        slots = slot_np[b]
        k_modes = means_np.shape[2]
        for mode_k in range(k_modes):
            pk = float(mix_probs_np[b, n, mode_k])
            for t in range(pred_len):
                if not bool(fut_valid[b, t, n]):
                    continue
                k = int(slots[obs_len + t])
                x = float(means_np[b, n, mode_k, t, 0])
                y = float(means_np[b, n, mode_k, t, 1])
                if t == 0:
                    lx, ly = float(obs_xy[b, obs_len - 1, n, 0]), float(obs_xy[b, obs_len - 1, n, 1])
                    vx = (x - lx) * hz
                    vy = (y - ly) * hz
                else:
                    px = float(means_np[b, n, mode_k, t - 1, 0])
                    py = float(means_np[b, n, mode_k, t - 1, 1])
                    vx = (x - px) * hz
                    vy = (y - py) * hz
                hd = math.atan2(vy, vx + 1e-8) if (vx != 0 or vy != 0) else 0.0
                rows.append(
                    _base_row(
                        scene_id=sid,
                        sequence_id=sq,
                        agent_id=int(agent_id_np[b, n]),
                        agent_class=agent_class,
                        raw_label=raw_s,
                        mapped_label=map_s,
                        frame_index=k,
                        time_index=k,
                        sample_rate_hz=hz,
                        phase=PHASE_PREDICTED,
                        mode_id=int(mode_k),
                        mode_probability=pk,
                        x=x,
                        y=y,
                        vx=float(vx),
                        vy=float(vy),
                        heading_rad=float(hd),
                        is_prediction=True,
                        is_deterministic=False,
                        source=SOURCE_MODEL,
                        coordinate_mode=coordinate_mode,
                        transform=transform,
                        export_run_id=export_run_id,
                        track_valid=bool(fut_valid[b, t, n]),
                        confidence=pk,
                    )
                )

    unknown = [m for m in modes if m not in ("deterministic", "multimodal", "ground_truth", "combined")]
    if unknown:
        raise ValueError(f"Unknown carla_export.export_modes entries: {unknown}")

    for b in range(B):
        for n in range(n_max):
            if not _agent_is_real(int(agent_id_np[b, n]), int(class_idx_np[b, n])):
                continue
            emit_obs(b, n)
            emit_gt(b, n)
            emit_pred_det(b, n)
            emit_pred_mm(b, n)

    return rows


def sort_export_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable sort per ``docs/carla_export.md``."""
    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(r["scene_id"]),
            str(r["sequence_id"]),
            int(r["agent_id"]),
            PHASE_SORT_ORDER.get(str(r["phase"]), 99),
            int(r["time_index"]),
            int(r["frame_index"]),
            int(r["mode_id"]),
        )

    return sorted(rows, key=key)


def write_carla_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    float_format: str | None = "%.6f",
    sort_rows: bool = True,
) -> None:
    """Write rows to CSV; float columns formatted via pandas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        df = pd.DataFrame(columns=default_column_order())
        df.to_csv(path, index=False)
        return
    if sort_rows:
        rows = sort_export_rows(rows)
    df = pd.DataFrame(rows)
    cols = [c for c in default_column_order() if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    df.to_csv(path, index=False, float_format=float_format)
