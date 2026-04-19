#!/usr/bin/env python3
"""Save trajectory PNGs from CHG-Net predictions (Phase 10 debug plots)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chgnet.datasets import SDDProcessedDataset, collate_sdd_batch
from chgnet.models.chg_net import chg_net_from_config
from chgnet.trainers.chg_trainer import load_model_weights
from chgnet.utils.config import load_config
from chgnet.utils.logger import setup_logger
from chgnet.utils.seed import seed_everything
from chgnet.viz.trajectory_plot import plot_batch_item_trajectories


def main() -> None:
    p = argparse.ArgumentParser(description="Plot obs / GT / pred trajectories (Matplotlib).")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override outputs.figures or visualization.output_dir.",
    )
    p.add_argument("--num-batches", type=int, default=1, help="Number of DataLoader batches to draw.")
    p.add_argument(
        "--items-per-batch",
        type=int,
        default=4,
        help="Max samples per batch to save as PNG (indices 0..B-1).",
    )
    p.add_argument("extra_configs", nargs="*", type=Path)
    args = p.parse_args()

    log = setup_logger(__name__)
    cfg = load_config(args.config, *args.extra_configs)
    seed_everything(
        int(cfg["experiment"]["seed"]),
        deterministic_torch=bool(cfg["experiment"].get("deterministic_torch", False)),
    )

    out = cfg.get("outputs", {})
    if not isinstance(out, dict):
        out = {}
    viz = cfg.get("visualization", {})
    if not isinstance(viz, dict):
        viz = {}
    fig_dir = Path(
        args.output_dir
        or viz.get("output_dir")
        or out.get("figures", "outputs/figures")
    )

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

    fig_dir.mkdir(parents=True, exist_ok=True)
    global_idx = 0
    batches_done = 0
    for batch in loader:
        batch_d = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch_d)

        B = batch["obs_xy"].shape[0]
        n_items = min(args.items_per_batch, B)
        for i in range(n_items):
            path = fig_dir / f"traj_{global_idx:05d}.png"
            plot_batch_item_trajectories(batch, cfg, i, out, out_path=path)
            global_idx += 1
        batches_done += 1
        log.info("Wrote %s samples from batch %s", n_items, batches_done)
        if batches_done >= args.num_batches:
            break

    log.info("Done. Figures under %s", fig_dir.resolve())


if __name__ == "__main__":
    main()
