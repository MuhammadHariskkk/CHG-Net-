#!/usr/bin/env python3
"""Export CARLA CSV rows from a trained CHG-Net and preprocessed SDD (Phase 9)."""

from __future__ import annotations

import argparse
import uuid
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chgnet.datasets import SDDProcessedDataset, collate_sdd_batch
from chgnet.exporters.carla_csv import build_export_rows, safe_scene_filename, write_carla_csv
from chgnet.models.chg_net import chg_net_from_config
from chgnet.trainers.chg_trainer import load_model_weights
from chgnet.utils.config import load_config
from chgnet.utils.logger import setup_logger
from chgnet.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description="Export CARLA CSV from CHG-Net predictions.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint (.pt).")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Which processed split to export (data.splits).",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Override carla_export.output_dir.")
    p.add_argument("--max-batches", type=int, default=None, help="Stop after N batches (debug).")
    p.add_argument("extra_configs", nargs="*", type=Path)
    args = p.parse_args()

    log = setup_logger(__name__)
    cfg = load_config(args.config, *args.extra_configs)

    seed_everything(
        int(cfg["experiment"]["seed"]),
        deterministic_torch=bool(cfg["experiment"].get("deterministic_torch", False)),
    )

    ce = cfg.get("carla_export", {})
    if not isinstance(ce, dict):
        ce = {}
    out_root = Path(args.output_dir or ce.get("output_dir", "outputs/carla_export"))
    layout = str(ce.get("file_layout", "one_scene_per_file"))
    sort_rows = bool(ce.get("sort_rows", True))
    float_fmt = ce.get("float_format", "%.6f")
    if float_fmt is not None:
        float_fmt = str(float_fmt)

    data = cfg["data"]
    root = Path(data["processed_root"])
    bs = int(cfg.get("inference", {}).get("batch_size", cfg["training"].get("batch_size", 64)))
    nw = int(cfg.get("inference", {}).get("num_workers", data.get("num_workers", 0)))
    pin_memory = bool(data.get("pin_memory", True))

    ds = SDDProcessedDataset(root, cfg, split=args.split)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collate_sdd_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = chg_net_from_config(cfg)
    load_model_weights(model, args.checkpoint, map_location=device)
    model.eval()

    export_run_id = str(uuid.uuid4())
    log.info("export_run_id=%s device=%s layout=%s", export_run_id, device, layout)

    out_root.mkdir(parents=True, exist_ok=True)

    by_scene: dict[str, list] = defaultdict(list)
    batch_idx = 0
    for batch in loader:
        batch_idx += 1
        batch_d = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch_d)
        rows = build_export_rows(batch, out, cfg, export_run_id=export_run_id)

        if layout == "one_batch_per_file":
            path = out_root / f"export_batch_{batch_idx:05d}.csv"
            write_carla_csv(rows, path, float_format=float_fmt, sort_rows=sort_rows)
        elif layout == "single_combined":
            by_scene["__all__"].extend(rows)
        elif layout == "one_scene_per_file":
            for row in rows:
                by_scene[str(row["scene_id"])].append(row)
        else:
            raise ValueError(f"Unknown carla_export.file_layout: {layout!r}")

        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

    if layout == "single_combined":
        path = out_root / "carla_export.csv"
        write_carla_csv(by_scene["__all__"], path, float_format=float_fmt, sort_rows=sort_rows)
    elif layout == "one_scene_per_file":
        for sid, rlist in by_scene.items():
            path = out_root / f"{safe_scene_filename(sid)}.csv"
            write_carla_csv(rlist, path, float_format=float_fmt, sort_rows=sort_rows)

    log.info("Wrote CARLA CSV under %s", out_root.resolve())


if __name__ == "__main__":
    main()
