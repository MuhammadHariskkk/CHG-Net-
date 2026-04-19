# CARLA CSV export (planned: Phase 9)

## Purpose

**Primary** downstream visualization path: CSV files for a separate CARLA playback pipeline. Matplotlib remains **secondary** debug-only (**Phase 10**).

## Required columns (contract)

Minimum columns (see user spec):  
`scene_id`, `sequence_id`, `agent_id`, `agent_class`, `raw_label`, `mapped_label`, `frame_index`, `time_index`, `timestamp_sec`, `phase` (`observed` | `ground_truth` | `predicted`), `mode_id`, `mode_probability`, `x`, `y`, `vx`, `vy`, `heading_rad`, `is_prediction`, `is_deterministic`, `source` (`dataset` | `model`).

**Engineering assumption (column semantics):** `mapped_label` is the canonical internal taxonomy string (e.g. `pedestrian`). `agent_class` must use the **same** string as `mapped_label` on every row so downstream CARLA tooling can rely on either column; they are not allowed to diverge in Phase 9 exports.

Preferred optional columns:  
`scene_name`, `sample_rate_hz`, `world_x`, `world_y`, `confidence`, `track_valid`, `export_run_id`.

## Export modes

1. **deterministic** — single selected mode (highest mixture weight; **paper-specified** inference default).
2. **multimodal** — all *K* modes with probabilities.
3. **ground_truth** — optional GT trajectory rows in the same schema.
4. **combined** — observed + GT + prediction in one deterministic row order.

## Coordinate policy

- **`raw`**: dataset image-plane coordinates (default when no mapping).
- **`world`**: optional pluggable transform from config (`carla_export.world_transform_path`); **Phase 9** implements transform hooks; without metadata, exports stay **raw** consistently.

## Timestamp policy

- Default `sample_rate_hz` = **2.5** from config; `timestamp_sec` derived from `time_index / sample_rate_hz` unless overridden.

## Reproducibility

- When `carla_export.sort_rows` is true, sort keys are: `scene_id`, `sequence_id`, `agent_id`, `phase`, `time_index`, `frame_index`, `mode_id` (tie-breakers as needed for stable CSV diffs).
- `configs/default.yaml` → `carla_export.export_modes` is a **YAML list** of mode names (e.g. `[deterministic]` or `[deterministic, multimodal]`); a single string would be ambiguous for multi-mode runs.
- Mode IDs and probabilities align with GMM head output tensor layout (**Phase 6–7**).

---

*Phase 9 adds `chgnet/exporters/schema.py`, `carla_csv.py`, and `scripts/export_carla_csv.py` with full API signatures.*
