"""Shared fixtures for CHG-Net tests (Phase 11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.helpers import make_preprocessed_sample


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_cfg(repo_root: Path) -> dict[str, Any]:
    from chgnet.utils.config import load_config

    return load_config(repo_root / "configs" / "default.yaml")


@pytest.fixture
def synthetic_batch(default_cfg: dict[str, Any]) -> dict[str, Any]:
    from chgnet.datasets.collate import collate_sdd_batch

    data = default_cfg["data"]
    obs_len = int(data["obs_len"])
    pred_len = int(data["pred_len"])
    s1 = make_preprocessed_sample(obs_len=obs_len, pred_len=pred_len, n_agents=2, scene_id="a", sequence_id="1")
    s2 = make_preprocessed_sample(obs_len=obs_len, pred_len=pred_len, n_agents=3, scene_id="b", sequence_id="2")
    return collate_sdd_batch([s1, s2])
