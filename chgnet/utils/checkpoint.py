"""Checkpoint save/load for PyTorch training (reusable by trainer in later phases)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    epoch: int | None = None,
    best_metric: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write a checkpoint ``.pt`` file (state dicts and optional metadata).

    **Engineering assumption**: write to a temporary file in the same directory then replace,
    to avoid corrupt partial files on interruption.

    Args:
        path: Destination path (``.pt`` recommended).
        model_state: ``model.state_dict()``.
        optimizer_state: Optional ``optimizer.state_dict()``.
        scheduler_state: Optional ``scheduler.state_dict()``.
        epoch: Training epoch index (optional).
        best_metric: Validation metric for best-model tracking (optional).
        extra: Arbitrary picklable metadata (config dict, git hash, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "epoch": epoch,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load checkpoint from disk. Returns the full payload dict.

    Args:
        path: Checkpoint path.
        map_location: Passed to ``torch.load`` (e.g. ``"cpu"`` or ``"cuda:0"``).

    Returns:
        Dictionary with keys ``model_state``, ``optimizer_state``, etc., as saved.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
    # PyTorch 2.6+ defaults weights_only=True; full checkpoints need False. Older 2.0.x
    # builds may not accept weights_only (TypeError) — branch on the live signature.
    load_kw: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    return torch.load(path, **load_kw)
