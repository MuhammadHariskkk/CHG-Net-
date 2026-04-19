from __future__ import annotations

from pathlib import Path


def test_load_default_yaml(repo_root: Path) -> None:
    from chgnet.utils.config import load_config

    cfg = load_config(repo_root / "configs" / "default.yaml")
    assert cfg["data"]["obs_len"] == 8
    assert cfg["data"]["pred_len"] == 12
    assert "labels" in cfg
    assert "training" in cfg
    assert "carla_export" in cfg
