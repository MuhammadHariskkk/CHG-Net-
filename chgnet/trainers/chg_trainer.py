"""Training and validation loop for :class:`~chgnet.models.chg_net.CHGNet`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from chgnet.losses.gmm_nll import gmm_trajectory_nll
from chgnet.metrics.trajectory import ade_fde_deterministic
from chgnet.models.chg_net import CHGNet
from chgnet.utils.checkpoint import load_checkpoint, save_checkpoint


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _build_optimizer(cfg: Mapping[str, Any], params: Any) -> torch.optim.Optimizer:
    t = cfg["training"]
    name = str(t.get("optimizer", "adam")).lower()
    lr = float(t["learning_rate"])
    wd = float(t.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown training.optimizer: {name!r} (use 'adam' or 'adamw').")


class CHGTrainer:
    """Fit ``CHGNet`` with GMM NLL; validate ADE/FDE on deterministic predictions."""

    def __init__(
        self,
        model: CHGNet,
        cfg: Mapping[str, Any],
        *,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        t = cfg["training"]
        self._max_epochs = int(t["max_epochs"])
        self._grad_clip = float(t.get("grad_clip_norm", 0.0))
        self._log_every = int(t.get("log_every_n_steps", 50))
        self._val_every = int(t.get("val_every_n_epochs", 1))
        self._save_best_only = bool(t.get("save_best_only", True))
        self._ckpt_dir = Path(t.get("checkpoint_dir", "outputs/checkpoints"))
        self._hist_dir = Path(t.get("history_dir", "outputs/history"))
        self._metric_name = str(t.get("best_checkpoint_metric", "val_ade"))
        self._lower_better = bool(t.get("best_checkpoint_lower_is_better", True))

        self.optimizer = _build_optimizer(cfg, self.model.parameters())
        self._global_step = 0
        self._history: list[dict[str, Any]] = []
        self._best_metric: float | None = None

    def _is_better(self, value: float) -> bool:
        if self._best_metric is None:
            return True
        if self._lower_better:
            return value < self._best_metric
        return value > self._best_metric

    def train_step(self, batch: Mapping[str, Any]) -> float:
        self.model.train()
        batch_d = _move_batch_to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        out = self.model(batch_d)
        loss = gmm_trajectory_nll(
            out.gmm,
            batch_d["fut_xy"],
            node_mask=out.node_mask,
            fut_valid=batch_d["fut_valid"],
        )
        loss.backward()
        if self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
        self.optimizer.step()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def eval_batch(self, batch: Mapping[str, Any]) -> tuple[float, float, float]:
        self.model.eval()
        batch_d = _move_batch_to_device(batch, self.device)
        out = self.model(batch_d)
        nll = gmm_trajectory_nll(
            out.gmm,
            batch_d["fut_xy"],
            node_mask=out.node_mask,
            fut_valid=batch_d["fut_valid"],
        )
        ade, fde = ade_fde_deterministic(
            out.gmm.deterministic_trajectory,
            batch_d["fut_xy"],
            node_mask=out.node_mask,
            fut_valid=batch_d["fut_valid"],
        )
        return float(nll.cpu()), float(ade.cpu()), float(fde.cpu())

    def train_epoch(self, epoch: int) -> float:
        total = 0.0
        n_batches = 0
        for batch in self.train_loader:
            self._global_step += 1
            loss = self.train_step(batch)
            total += loss
            n_batches += 1
            if self._log_every > 0 and self._global_step % self._log_every == 0:
                print(f"epoch {epoch} step {self._global_step} train_nll {loss:.6f}")
        return total / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        sum_nll = sum_ade = sum_fde = 0.0
        n_batches = 0
        for batch in self.val_loader:
            nv, ade, fde = self.eval_batch(batch)
            sum_nll += nv
            sum_ade += ade
            sum_fde += fde
            n_batches += 1
        if n_batches == 0:
            return {}
        return {
            "val_nll": sum_nll / n_batches,
            "val_ade": sum_ade / n_batches,
            "val_fde": sum_fde / n_batches,
        }

    def _metric_value(self, metrics: Mapping[str, float]) -> float | None:
        key = self._metric_name
        if key not in metrics:
            return None
        return float(metrics[key])

    def fit(self, start_epoch: int = 0) -> list[dict[str, Any]]:
        """Run training; optionally resume ``start_epoch`` from checkpoint metadata."""
        self._hist_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, self._max_epochs):
            train_loss = self.train_epoch(epoch + 1)
            row: dict[str, Any] = {"epoch": epoch + 1, "train_nll": train_loss}

            if self.val_loader is not None and (epoch + 1) % self._val_every == 0:
                vm = self.validate()
                row.update(vm)
                mv = self._metric_value(vm)
                if mv is not None and self._is_better(mv):
                    self._best_metric = mv
                    save_checkpoint(
                        self._ckpt_dir / "checkpoint_best.pt",
                        model_state=self.model.state_dict(),
                        optimizer_state=self.optimizer.state_dict(),
                        epoch=epoch + 1,
                        best_metric=mv,
                        extra={"cfg_keys": list(self.cfg.keys()), "metric": self._metric_name},
                    )

            self._history.append(row)
            hist_path = self._hist_dir / "train_history.json"
            hist_path.write_text(json.dumps(self._history, indent=2), encoding="utf-8")

            if not self._save_best_only:
                save_checkpoint(
                    self._ckpt_dir / "checkpoint_last.pt",
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    epoch=epoch + 1,
                    best_metric=self._best_metric,
                    extra={"cfg_keys": list(self.cfg.keys())},
                )

        return self._history

    def load_training_state(self, payload: Mapping[str, Any]) -> int:
        """Restore optimizer and best-metric tracker; return 0-based index for :meth:`fit` ``start_epoch``."""
        opt = payload.get("optimizer_state")
        if opt is not None:
            self.optimizer.load_state_dict(opt)
        self._best_metric = payload.get("best_metric")
        return int(payload.get("epoch") or 0)


def load_model_weights(model: nn.Module, path: str | Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Load ``model_state`` from a checkpoint saved by :func:`chgnet.utils.checkpoint.save_checkpoint`."""
    payload = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    return payload
