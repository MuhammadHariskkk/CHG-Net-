"""Reproducibility helpers for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Set seeds for ``random``, ``numpy``, and ``torch`` (CPU and CUDA if available).

    **Engineering assumption**: ``deterministic_torch=True`` enables CUDNN deterministic algorithms
    where supported and may reduce throughput; use for strict reproducibility debugging.

    Args:
        seed: Integer seed.
        deterministic_torch: If True, set ``torch.backends.cudnn.deterministic`` and disable
            benchmark mode for more reproducible GPU results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
