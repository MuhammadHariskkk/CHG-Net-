"""Stanford Drone Dataset preprocessing: annotations → resampled windows + metadata.

**paper-specified**: ``obs_len``, ``pred_len``, ``sample_rate_hz`` from config (defaults 8, 12, 2.5).

**engineering assumption**:
- ``annotations.txt`` layout: track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
  (Stanford Drone Dataset convention; label may contain spaces inside quotes).
- Native timeline resampled with ``source_fps`` (default 30) to ``data.sample_rate_hz``.
- Nearest-frame lookup for each resampled grid index (no linear interpolation).
- **Window agent set:** if ``require_full_window_valid`` (default true), agents must be valid on every obs
  and future step; otherwise agents valid only at the last observation step are included. Per-step
  validity is stored in ``obs_valid`` / ``fut_valid`` (see ``build_samples_for_sequence``).
- Heading uses atan2(vy, vx); if speed < ``zero_speed_eps``, reuse last heading then 0 (documented below).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from chgnet.datasets.label_mapping import LabelMapper, build_label_mapper_from_config, normalize_raw_label
from chgnet.datasets.scene_split import get_scene_splits

# Heading / speed (**engineering assumption**; matches docs/sdd_preprocessing.md).
_ZERO_SPEED_EPS: float = 1e-3


@dataclass
class AnnotationRow:
    track_id: int
    frame: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    lost: int
    occluded: int
    generated: int
    label: str


def parse_annotations_txt(path: Path) -> list[AnnotationRow]:
    """Parse a single SDD-style ``annotations.txt`` file."""
    rows: list[AnnotationRow] = []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        label = ""
        head_str = line
        if '"' in line:
            first = line.find('"')
            last = line.rfind('"')
            if last > first:
                label = line[first + 1 : last]
                head_str = line[:first].strip()
        parts = head_str.split()
        if label == "":
            if len(parts) < 10:
                continue
            parts_all = line.split()
            if len(parts_all) < 10:
                continue
            head = parts_all[:9]
            label = " ".join(parts_all[9:]).strip().strip('"').strip("'")
        else:
            if len(parts) < 9:
                continue
            head = parts[:9]
        try:
            tid = int(head[0])
            xmin, ymin, xmax, ymax = map(float, head[1:5])
            frame = int(head[5])
            lost, occluded, generated = map(int, head[6:9])
        except (ValueError, IndexError):
            continue
        rows.append(
            AnnotationRow(
                track_id=tid,
                frame=frame,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                lost=lost,
                occluded=occluded,
                generated=generated,
                label=label,
            )
        )
    return rows


def discover_annotation_files(raw_root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(scene_id, sequence_id, annotations_path)``.

    Supports ``<scene>/<video>/annotations.txt`` or ``<scene>/annotations.txt`` (sequence_id ``_root``).
    """
    if not raw_root.is_dir():
        return
    for scene_dir in sorted(raw_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        found_video = False
        for sub in sorted(scene_dir.iterdir()):
            if not sub.is_dir():
                continue
            ann = sub / "annotations.txt"
            if ann.is_file():
                found_video = True
                yield scene_id, sub.name, ann
        if not found_video:
            ann = scene_dir / "annotations.txt"
            if ann.is_file():
                yield scene_id, "_root", ann


def _center_xy(row: AnnotationRow) -> tuple[float, float]:
    return (0.5 * (row.xmin + row.xmax), 0.5 * (row.ymin + row.ymax))


def _in_bounds(xy: tuple[float, float], bounds: list[float] | None) -> bool:
    if bounds is None or len(bounds) != 4:
        return True
    min_x, min_y, max_x, max_y = bounds
    x, y = xy
    return min_x <= x <= max_x and min_y <= y <= max_y


@dataclass
class TrackSeries:
    track_id: int
    label_raw: str
    frames: np.ndarray  # int32, sorted unique raw frames
    xy: np.ndarray  # float32 (T, 2)
    valid: np.ndarray  # bool (T,) — not lost, in bounds


def build_track_series(
    rows: list[AnnotationRow],
    bounds: list[float] | None,
    filter_invalid_coords: bool,
    drop_stats: DropStats | None,
) -> dict[int, TrackSeries]:
    """Aggregate rows into per-track time series on the native frame grid (last row wins on duplicates)."""
    by_track: dict[int, list[AnnotationRow]] = defaultdict(list)
    for r in rows:
        by_track[r.track_id].append(r)
    out: dict[int, TrackSeries] = {}
    for tid, rlist in by_track.items():
        rlist.sort(key=lambda z: z.frame)
        frames_list: list[int] = []
        xy_list: list[tuple[float, float]] = []
        val_list: list[bool] = []
        label_raw_track = ""
        for r in rlist:
            xy = _center_xy(r)
            finite = math.isfinite(xy[0]) and math.isfinite(xy[1])
            if not finite:
                if drop_stats is not None:
                    drop_stats.invalid_coords += 1
                continue
            in_b = _in_bounds(xy, bounds)
            if filter_invalid_coords and not in_b:
                if drop_stats is not None:
                    drop_stats.invalid_coords += 1
                continue
            if not filter_invalid_coords and not in_b:
                # Keep point for continuity but mark invalid when out of bounds.
                ok = False
            else:
                ok = r.lost == 0 and in_b
            frames_list.append(int(r.frame))
            xy_list.append(xy)
            val_list.append(ok)
            label_raw_track = r.label
        if not frames_list:
            continue
        # Collapse duplicate frames: keep last occurrence (list is sorted by frame).
        f_np = np.array(frames_list, dtype=np.int32)
        xy_np = np.array(xy_list, dtype=np.float32)
        v_np = np.array(val_list, dtype=bool)
        uniq_frames: list[int] = []
        uniq_xy: list[np.ndarray] = []
        uniq_v: list[bool] = []
        last_f = None
        for i, f in enumerate(f_np):
            if last_f is None or f != last_f:
                uniq_frames.append(int(f))
                uniq_xy.append(xy_np[i])
                uniq_v.append(bool(v_np[i]))
                last_f = f
            else:
                uniq_xy[-1] = xy_np[i]
                uniq_v[-1] = bool(v_np[i])
        frames_arr = np.array(uniq_frames, dtype=np.int32)
        xy_arr = np.vstack(uniq_xy).astype(np.float32)
        valid_arr = np.array(uniq_v, dtype=bool)
        out[tid] = TrackSeries(
            track_id=tid,
            label_raw=label_raw_track,
            frames=frames_arr,
            xy=xy_arr,
            valid=valid_arr,
        )
    return out


def resample_track_to_grid(
    series: TrackSeries,
    frame_min: int,
    frame_max: int,
    source_fps: float,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map native frames to resampled indices k = 0..K-1 where slot k centers at frame_min + k * ratio.

    Returns ``(xy_k, valid_k, label_per_k str as object array)`` with shape (K, 2), (K,), (K,).
    **Engineering assumption**: ratio = source_fps / sample_rate_hz; nearest native sample to target frame.
    """
    ratio = float(source_fps) / float(sample_rate_hz)
    if ratio <= 0:
        raise ValueError("source_fps and sample_rate_hz must be positive.")
    k_max = int(np.floor((frame_max - frame_min) / ratio))
    if k_max < 0:
        k_max = 0
    k_count = k_max + 1
    xy_out = np.full((k_count, 2), np.nan, dtype=np.float32)
    valid_out = np.zeros((k_count,), dtype=bool)
    frames = series.frames
    xy = series.xy
    valid = series.valid
    if len(frames) == 0:
        return xy_out, valid_out, np.array([""] * k_count, dtype=object)
    for k in range(k_count):
        target_f = frame_min + k * ratio
        # nearest neighbor in sorted frames
        j = int(np.searchsorted(frames, target_f, side="left"))
        candidates = []
        if j < len(frames):
            candidates.append(j)
        if j - 1 >= 0:
            candidates.append(j - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(float(frames[idx]) - target_f))
        xy_out[k] = xy[best]
        valid_out[k] = valid[best]
    labels = np.array([series.label_raw] * k_count, dtype=object)
    return xy_out, valid_out, labels


def velocities_from_positions(xy: np.ndarray, valid: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Finite-difference velocities (per second) on the observation grid."""
    dt = 1.0 / float(sample_rate_hz)
    v = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[0] < 2:
        return v
    for t in range(1, xy.shape[0]):
        if valid[t] and valid[t - 1]:
            v[t] = (xy[t] - xy[t - 1]) / dt
    if xy.shape[0] >= 2:
        v[0] = v[1]
    return v


def headings_from_velocities(vel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Heading radians from velocity; zero-speed → last valid heading, else 0."""
    h = np.zeros((vel.shape[0],), dtype=np.float32)
    last_h = 0.0
    for t in range(vel.shape[0]):
        vx, vy = float(vel[t, 0]), float(vel[t, 1])
        speed = math.hypot(vx, vy)
        if not valid[t] or speed < _ZERO_SPEED_EPS:
            h[t] = last_h
            continue
        last_h = float(math.atan2(vy, vx))
        h[t] = last_h
    return h


def relative_displacement_to_last(xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """For each obs step t: xy[t] - xy[T-1] (last observed); invalid steps keep NaN where invalid."""
    out = np.full_like(xy, np.nan, dtype=np.float32)
    t_last = xy.shape[0] - 1
    ref = xy[t_last]
    if not valid[t_last]:
        return out
    for t in range(xy.shape[0]):
        if valid[t]:
            out[t] = xy[t] - ref
    return out


@dataclass
class SceneStats:
    raw_counts: Counter[str] = field(default_factory=Counter)
    mapped_counts: Counter[str] = field(default_factory=Counter)
    windows_emitted: int = 0
    windows_dropped_short: int = 0
    tracks_dropped_short: int = 0


@dataclass
class DropStats:
    too_short_native: int = 0
    invalid_coords: int = 0


def _labels_for_tracks(
    track_ids: list[int],
    track_series: dict[int, TrackSeries],
    mapper: LabelMapper,
) -> tuple[list[str], list[str], np.ndarray]:
    raw_labels: list[str] = []
    mapped_labels: list[str] = []
    idxs: list[int] = []
    for tid in track_ids:
        s = track_series[tid]
        _, model_c, ix = mapper.map_raw(s.label_raw)
        raw_labels.append(s.label_raw)
        mapped_labels.append(model_c)
        idxs.append(ix)
    return raw_labels, mapped_labels, np.array(idxs, dtype=np.int64)


def build_samples_for_sequence(
    scene_id: str,
    sequence_id: str,
    rows: list[AnnotationRow],
    cfg: Mapping[str, Any],
    mapper: LabelMapper,
    scene_stats: SceneStats,
    drop_stats: DropStats,
    raw_pair_counter: Counter[str],
) -> list[dict[str, Any]]:
    """Create sliding-window samples for one video sequence."""
    data = cfg["data"]
    obs_len = int(data["obs_len"])
    pred_len = int(data["pred_len"])
    sample_rate_hz = float(data["sample_rate_hz"])
    source_fps = float(data.get("source_fps", 30.0))
    bounds = data.get("image_bounds")
    stride = int(data.get("window_stride", 1))
    require_full = bool(data.get("require_full_window_valid", True))
    filter_invalid = bool(data.get("filter_invalid_coords", True))

    tracks = build_track_series(rows, bounds, filter_invalid, drop_stats)
    if not tracks:
        return []

    frame_min = min(int(s.frames.min()) for s in tracks.values())
    frame_max = max(int(s.frames.max()) for s in tracks.values())
    ratio = source_fps / sample_rate_hz
    k_total = int(np.floor((frame_max - frame_min) / ratio)) + 1
    need = obs_len + pred_len
    if k_total < need:
        drop_stats.too_short_native += 1
        return []

    # Resample all tracks to common K
    resampled: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for tid, ser in tracks.items():
        xy_k, v_k, _lab_k = resample_track_to_grid(ser, frame_min, frame_max, source_fps, sample_rate_hz)
        if xy_k.shape[0] < need:
            drop_stats.too_short_native += 1
            scene_stats.tracks_dropped_short += 1
            continue
        resampled[tid] = (xy_k, v_k, _lab_k)

    if not resampled:
        return []

    samples: list[dict[str, Any]] = []
    max_k0 = k_total - need
    for k0 in range(0, max_k0 + 1, stride):
        k_obs_end = k0 + obs_len - 1
        k_fut_end = k0 + need - 1
        slot_indices = np.arange(k0, k0 + need, dtype=np.int32)

        # Agent set: tracks with valid obs at k_obs_end and all obs slots valid if require_full
        active: list[int] = []
        for tid, (xy_k, val_k, _) in resampled.items():
            if require_full:
                obs_slice = val_k[k0 : k0 + obs_len]
                fut_slice = val_k[k0 + obs_len : k0 + need]
                if obs_slice.size < obs_len or fut_slice.size < pred_len:
                    continue
                if not bool(obs_slice.all() and fut_slice.all()):
                    continue
            else:
                if not bool(val_k[k_obs_end]):
                    continue
            active.append(tid)
        active.sort()
        if not active:
            scene_stats.windows_dropped_short += 1
            continue

        n = len(active)
        obs_xy = np.full((obs_len, n, 2), np.nan, dtype=np.float32)
        fut_xy = np.full((pred_len, n, 2), np.nan, dtype=np.float32)
        obs_valid = np.zeros((obs_len, n), dtype=bool)
        fut_valid = np.zeros((pred_len, n), dtype=bool)

        for a, tid in enumerate(active):
            xy_k, val_k, _ = resampled[tid]
            for t in range(obs_len):
                k = k0 + t
                obs_xy[t, a] = xy_k[k]
                obs_valid[t, a] = val_k[k]
            for t in range(pred_len):
                k = k0 + obs_len + t
                fut_xy[t, a] = xy_k[k]
                fut_valid[t, a] = val_k[k]

        obs_vel = np.stack(
            [velocities_from_positions(obs_xy[:, a, :], obs_valid[:, a], sample_rate_hz) for a in range(n)],
            axis=1,
        )
        obs_heading = np.stack(
            [headings_from_velocities(obs_vel[:, a, :], obs_valid[:, a]) for a in range(n)],
            axis=1,
        )
        obs_rel = np.stack(
            [relative_displacement_to_last(obs_xy[:, a, :], obs_valid[:, a]) for a in range(n)],
            axis=1,
        )

        raw_labs, map_labs, cls_idx = _labels_for_tracks(active, tracks, mapper)

        for rl, ml in zip(raw_labs, map_labs, strict=True):
            scene_stats.raw_counts[rl] += 1
            scene_stats.mapped_counts[ml] += 1
            raw_pair_counter[f"{normalize_raw_label(rl)}|||{ml}"] += 1

        sample = {
            "scene_id": scene_id,
            "sequence_id": sequence_id,
            "resampled_frame_min": int(frame_min),
            "window_k_start": int(k0),
            "window_k_end_obs": int(k_obs_end),
            "window_k_end_fut": int(k_fut_end),
            "slot_indices": slot_indices,
            "obs_xy": obs_xy,
            "fut_xy": fut_xy,
            "obs_vel": obs_vel,
            "obs_heading": obs_heading,
            "obs_rel_disp": obs_rel,
            "obs_valid": obs_valid,
            "fut_valid": fut_valid,
            "agent_id": np.array(active, dtype=np.int64),
            "raw_label": raw_labs,
            "mapped_label": map_labs,
            "class_idx": cls_idx,
        }
        samples.append(sample)
        scene_stats.windows_emitted += 1

    return samples


def run_preprocessing(cfg: Mapping[str, Any], raw_root: Path | None = None) -> dict[str, Any]:
    """Run full SDD preprocessing pipeline; write artifacts under ``data.processed_root``.

    Returns a summary dict with paths and counts.
    """
    data = cfg["data"]
    root = Path(raw_root if raw_root is not None else data["raw_root"])
    out_root = Path(data["processed_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    pre = data.get("preprocess", {})
    meta_name = pre.get("meta_filename", "preprocess_meta.json")
    index_name = pre.get("index_filename", "dataset_index.json")
    chunk_prefix = pre.get("samples_prefix", "samples_chunk_")
    chunk_size = int(pre.get("chunk_size", 0))

    mapper = build_label_mapper_from_config(cfg)
    scene_filter = data.get("scene_filter")
    filter_set = set(scene_filter) if scene_filter is not None else None

    global_raw = Counter()
    global_mapped = Counter()
    raw_pair_counter: Counter[str] = Counter()
    scene_stats_map: dict[str, SceneStats] = defaultdict(SceneStats)
    drop_stats = DropStats()

    all_samples: list[dict[str, Any]] = []

    for scene_id, sequence_id, ann_path in discover_annotation_files(root):
        if filter_set is not None and scene_id not in filter_set:
            continue
        rows = parse_annotations_txt(ann_path)
        ss = scene_stats_map[scene_id]
        samples = build_samples_for_sequence(
            scene_id, sequence_id, rows, cfg, mapper, ss, drop_stats, raw_pair_counter
        )
        all_samples.extend(samples)

    # Serialize chunks
    chunk_entries: list[dict[str, Any]] = []
    if chunk_size and chunk_size > 0:
        for i in range(0, len(all_samples), chunk_size):
            chunk = all_samples[i : i + chunk_size]
            fname = f"{chunk_prefix}{i // chunk_size:05d}.pt"
            torch.save({"samples": chunk}, out_root / fname)
            chunk_entries.append({"path": fname, "num_samples": len(chunk)})
    else:
        fname = f"{chunk_prefix}00000.pt"
        torch.save({"samples": all_samples}, out_root / fname)
        chunk_entries.append({"path": fname, "num_samples": len(all_samples)})

    for st in scene_stats_map.values():
        global_raw.update(st.raw_counts)
        global_mapped.update(st.mapped_counts)

    meta: dict[str, Any] = {
        "version": 1,
        "class_vocabulary": list(mapper.vocabulary),
        "class_to_index": {n: i for i, n in enumerate(mapper.vocabulary)},
        "taxonomy_mode": mapper.taxonomy_mode,
        "splits_snapshot": get_scene_splits(cfg),
        "raw_label_counts": dict(global_raw),
        "mapped_label_counts": dict(global_mapped),
        "raw_to_mapped_counts": dict(raw_pair_counter),
        "scene_stats": {
            sid: {
                "raw_label_counts": dict(st.raw_counts),
                "mapped_label_counts": dict(st.mapped_counts),
                "windows_emitted": st.windows_emitted,
                "windows_dropped_no_agent": st.windows_dropped_short,
                "tracks_dropped_short": st.tracks_dropped_short,
            }
            for sid, st in scene_stats_map.items()
        },
        "dropped_stats": {
            "too_short_sequences": drop_stats.too_short_native,
            "invalid_coords_filtered": drop_stats.invalid_coords,
        },
        "config_snapshot": {
            "data.obs_len": data.get("obs_len"),
            "data.pred_len": data.get("pred_len"),
            "data.sample_rate_hz": data.get("sample_rate_hz"),
            "data.source_fps": data.get("source_fps"),
            "labels.taxonomy_mode": cfg["labels"].get("taxonomy_mode"),
        },
    }

    if cfg["labels"].get("emit_label_stats", True):
        (out_root / meta_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    index_obj = {
        "version": 1,
        "meta_path": meta_name,
        "chunks": chunk_entries,
        "total_samples": len(all_samples),
        "sample_scene_ids": [str(s["scene_id"]) for s in all_samples],
        "sample_sequence_ids": [str(s["sequence_id"]) for s in all_samples],
    }
    (out_root / index_name).write_text(json.dumps(index_obj, indent=2), encoding="utf-8")

    return {
        "processed_root": str(out_root.resolve()),
        "num_samples": len(all_samples),
        "meta_path": str((out_root / meta_name).resolve()),
        "index_path": str((out_root / index_name).resolve()),
    }
