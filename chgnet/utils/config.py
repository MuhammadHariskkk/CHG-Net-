"""YAML configuration loading with deep merge and optional overrides.

**Paper-specified** defaults (8/12 frames, 2.5 Hz, LR, dims, K=3) are reflected in `configs/default.yaml`.
**Engineering assumption**: dot-path override syntax and merge precedence (later files win).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``. Nested dicts are merged; lists and
    scalars from ``override`` replace those in ``base``.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = cfg
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _parse_override(override: str) -> tuple[str, Any]:
    """Parse ``key.path=value``; value is YAML-loaded for bool/int/float/list semantics."""
    if "=" not in override:
        raise ValueError(f"Override must be 'key=value', got: {override!r}")
    key, raw = override.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Could not parse override value as YAML: {raw!r}") from e
    return key, parsed


def load_config(
    *config_paths: str | Path,
    overrides: list[str] | None = None,
    defaults_first: str | Path | None = None,
) -> dict[str, Any]:
    """Load one or more YAML files and merge left-to-right (later wins).

    If ``defaults_first`` is set, that file is loaded before ``config_paths`` (useful to always
    anchor on ``default.yaml`` then apply ``sdd.yaml``).

    ``overrides`` entries use dotted keys, e.g. ``training.batch_size=32``; values are parsed with
    ``yaml.safe_load`` so ``true``, numbers, and JSON-like lists work.

    Returns a plain ``dict`` (mutable). Callers may wrap in their own typed config if desired.
    """
    paths: list[Path] = []
    if defaults_first is not None:
        paths.append(Path(defaults_first))
    paths.extend(Path(p) for p in config_paths)

    merged: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path.resolve()}")
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(f"Root of YAML must be a mapping: {path}")
        merged = deep_merge(merged, data)

    if overrides:
        for ov in overrides:
            key, val = _parse_override(ov)
            _set_dotted(merged, key, val)

    return merged


def config_dir() -> Path:
    """Directory containing packaged default configs (repo ``configs/`` at project root)."""
    # Assumption: configs live next to package root when running from repo.
    return Path(__file__).resolve().parents[2] / "configs"
