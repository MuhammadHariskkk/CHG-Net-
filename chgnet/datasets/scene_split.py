"""Scene-level train/val/test splits for SDD (config-driven).

**engineering assumption**: default split lists in ``configs/default.yaml`` are not defined in the
CHG-Net paper; they follow the eight public SDD scene names and a reasonable hold-out layout.
Override ``data.splits`` in YAML to match your benchmark.
"""

from __future__ import annotations

from typing import Any, Mapping


def get_scene_splits(cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return split name → list of top-level scene directory names."""
    data = cfg["data"]
    splits = data.get("splits")
    if splits is None:
        return {"train": [], "val": [], "test": []}
    return {k: list(v) for k, v in splits.items()}


def scene_in_split(scene_id: str, split: str, cfg: Mapping[str, Any]) -> bool:
    """True if ``scene_id`` is listed under ``data.splits[split]``, or if splits are unset (all scenes)."""
    allowed = scenes_for_split(split, cfg)
    if allowed is None:
        return True
    return scene_id in allowed


def scenes_for_split(split: str, cfg: Mapping[str, Any]) -> set[str] | None:
    """Set of scene ids for ``split``, or ``None`` if ``data.splits`` is unset/empty (no filtering).

    **Backward compatibility:** older configs without ``data.splits`` still load all processed samples.
    """
    splits = get_scene_splits(cfg)
    if not splits or all(len(v) == 0 for v in splits.values()):
        return None
    return set(splits.get(split, []))
