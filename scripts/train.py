#!/usr/bin/env python3
"""Train CHG-Net on preprocessed SDD (Phase 8)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chgnet.datasets import SDDProcessedDataset, collate_sdd_batch
from chgnet.models.chg_net import chg_net_from_config
from chgnet.trainers.chg_trainer import CHGTrainer
from chgnet.utils.checkpoint import load_checkpoint
from chgnet.utils.config import load_config
from chgnet.utils.logger import setup_logger
from chgnet.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description="Train CHG-Net (GMM NLL, val ADE/FDE).")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config (merged left-to-right if repeated).",
    )
    p.add_argument("--resume", type=Path, default=None, help="Optional checkpoint .pt to resume.")
    p.add_argument("extra_configs", nargs="*", type=Path, help="Additional YAML files to merge.")
    args = p.parse_args()

    log = setup_logger(__name__)
    cfg_paths = [args.config, *args.extra_configs]
    cfg = load_config(*cfg_paths)

    seed_everything(
        int(cfg["experiment"]["seed"]),
        deterministic_torch=bool(cfg["experiment"].get("deterministic_torch", False)),
    )

    data = cfg["data"]
    root = Path(data["processed_root"])
    num_workers = int(data.get("num_workers", 0))
    pin_memory = bool(data.get("pin_memory", True))

    train_ds = SDDProcessedDataset(root, cfg, split="train")
    val_ds = SDDProcessedDataset(root, cfg, split="val")

    bs = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collate_sdd_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collate_sdd_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    model = chg_net_from_config(cfg)
    trainer = CHGTrainer(
        model,
        cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    start_epoch = 0
    if args.resume is not None:
        payload = load_checkpoint(args.resume, map_location=device)
        trainer.model.load_state_dict(payload["model_state"])
        start_epoch = trainer.load_training_state(payload)
        log.info("Resumed from %s; next fit() epoch index=%s", args.resume, start_epoch)

    trainer.fit(start_epoch=start_epoch)
    log.info("Training finished.")


if __name__ == "__main__":
    main()
