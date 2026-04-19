"""CARLA CSV column contract (see ``docs/carla_export.md``)."""

from __future__ import annotations

from typing import Any, Final, Mapping

# Required columns (exact header strings).
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "scene_id",
    "sequence_id",
    "agent_id",
    "agent_class",
    "raw_label",
    "mapped_label",
    "frame_index",
    "time_index",
    "timestamp_sec",
    "phase",
    "mode_id",
    "mode_probability",
    "x",
    "y",
    "vx",
    "vy",
    "heading_rad",
    "is_prediction",
    "is_deterministic",
    "source",
)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "scene_name",
    "sample_rate_hz",
    "world_x",
    "world_y",
    "confidence",
    "track_valid",
    "export_run_id",
)

PHASE_OBSERVED: Final[str] = "observed"
PHASE_GROUND_TRUTH: Final[str] = "ground_truth"
PHASE_PREDICTED: Final[str] = "predicted"

SOURCE_DATASET: Final[str] = "dataset"
SOURCE_MODEL: Final[str] = "model"

# Stable sort: phase order then time_index, frame_index, mode_id (docs/carla_export.md).
PHASE_SORT_ORDER: Final[dict[str, int]] = {
    PHASE_OBSERVED: 0,
    PHASE_GROUND_TRUTH: 1,
    PHASE_PREDICTED: 2,
}


def default_column_order() -> list[str]:
    """Headers: required first, then optional keys present in written rows (caller may filter)."""
    return list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)


def validate_row_minimal(row: Mapping[str, Any]) -> None:
    """Raise if a row dict is missing any required key."""
    missing = [c for c in REQUIRED_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"CARLA row missing required keys: {missing}")


def normalize_export_modes(raw: Any) -> list[str]:
    """Accept YAML list of mode names; a single string becomes a one-element list (backward compatible)."""
    if raw is None:
        return ["deterministic"]
    if isinstance(raw, str):
        return [raw.strip().lower()]
    if isinstance(raw, (list, tuple)):
        return [str(m).strip().lower() for m in raw]
    raise TypeError(f"carla_export.export_modes must be str or list, got {type(raw).__name__}")
