#!/usr/bin/env python
"""Preprocess raw Stanford Drone Dataset annotations into chunked torch samples + metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/preprocess_data.py` from repo root without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chgnet.utils.config import load_config
from chgnet.utils.logger import setup_logger
from chgnet.utils.seed import seed_everything
from chgnet.datasets.preprocessing import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="SDD preprocessing for CHG-Net")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=["configs/sdd.yaml"],
        help="YAML config paths merged left-to-right after defaults.",
    )
    parser.add_argument(
        "--defaults",
        type=str,
        default="configs/default.yaml",
        help="Base defaults YAML loaded first.",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Override data.raw_root for this run.",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help="Override data.processed_root for this run.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. data.scene_filter=[coupa]",
    )
    args = parser.parse_args()

    cfg_paths = [Path(p) for p in args.config]
    cwd = Path.cwd()
    resolved_defaults = args.defaults
    if not Path(resolved_defaults).is_file():
        alt = _REPO_ROOT / resolved_defaults
        if alt.is_file():
            resolved_defaults = str(alt)
    resolved_cfgs = []
    for p in cfg_paths:
        if not p.is_file():
            alt = _REPO_ROOT / p
            if alt.is_file():
                p = alt
        resolved_cfgs.append(p)

    cfg = load_config(
        *resolved_cfgs,
        defaults_first=resolved_defaults,
        overrides=args.override or None,
    )
    if args.raw_root is not None:
        cfg["data"]["raw_root"] = args.raw_root
    if args.processed_root is not None:
        cfg["data"]["processed_root"] = args.processed_root

    log = setup_logger(__name__)
    exp = cfg.get("experiment", {})
    seed_everything(int(exp.get("seed", 42)), deterministic_torch=bool(exp.get("deterministic_torch", False)))

    raw = Path(cfg["data"]["raw_root"])
    if not raw.is_dir():
        log.warning("Raw root %s is not a directory (empty run / dry check).", raw.resolve())

    summary = run_preprocessing(cfg, raw_root=raw)
    log.info("Wrote %d samples → %s", summary["num_samples"], summary["processed_root"])
    log.info("Meta: %s", summary["meta_path"])
    log.info("Index: %s", summary["index_path"])


if __name__ == "__main__":
    main()
