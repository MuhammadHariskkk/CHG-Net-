#!/usr/bin/env python
"""Print raw / mapped label distributions from preprocessing metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from chgnet.utils.config import load_config


def _resolve_path(p: str, repo_root: Path) -> Path:
    path = Path(p)
    if path.is_file():
        return path
    alt = repo_root / p
    if alt.is_file():
        return alt
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SDD label distributions (preprocess_meta.json)")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=["configs/sdd.yaml"],
        help="YAML configs (merged) to locate preprocess meta path.",
    )
    parser.add_argument(
        "--defaults",
        type=str,
        default="configs/default.yaml",
        help="Defaults YAML loaded first.",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Explicit path to preprocess_meta.json (overrides config).",
    )
    args = parser.parse_args()

    defaults = _resolve_path(args.defaults, _REPO_ROOT)
    cfg_paths = [_resolve_path(p, _REPO_ROOT) for p in args.config]
    cfg = load_config(*cfg_paths, defaults_first=str(defaults))

    if args.meta is not None:
        meta_path = Path(args.meta)
    else:
        root = Path(cfg["data"]["processed_root"])
        pre = cfg["data"].get("preprocess", {})
        meta_path = root / pre.get("meta_filename", "preprocess_meta.json")

    if not meta_path.is_file():
        raise SystemExit(f"Metadata not found: {meta_path.resolve()}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print("=== Global raw label counts ===")
    raw = pd.Series(meta.get("raw_label_counts", {}), name="count").sort_values(ascending=False)
    print(raw.to_string())
    print("\n=== Global mapped label counts ===")
    mapped = pd.Series(meta.get("mapped_label_counts", {}), name="count").sort_values(ascending=False)
    print(mapped.to_string())
    print("\n=== Per-scene mapped label counts ===")
    for scene, st in sorted(meta.get("scene_stats", {}).items()):
        print(f"-- {scene} --")
        m = pd.Series(st.get("mapped_label_counts", {}), name="count").sort_values(ascending=False)
        print(m.to_string())
        print()
    print("=== Raw → mapped (normalized raw key) edge counts (top 30) ===")
    edges = pd.Series(meta.get("raw_to_mapped_counts", {}), name="count").sort_values(ascending=False).head(30)
    print(edges.to_string())


if __name__ == "__main__":
    main()
