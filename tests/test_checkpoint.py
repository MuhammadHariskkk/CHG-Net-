from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from chgnet.models.chg_net import chg_net_from_config
from chgnet.utils.checkpoint import load_checkpoint, save_checkpoint


def test_save_load_checkpoint_roundtrip(default_cfg: dict[str, Any], tmp_path: Path) -> None:
    m = chg_net_from_config(default_cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    path = tmp_path / "t.pt"
    save_checkpoint(
        path,
        model_state=m.state_dict(),
        optimizer_state=opt.state_dict(),
        epoch=3,
        best_metric=1.23,
        extra={"k": "v"},
    )
    payload = load_checkpoint(path, map_location="cpu")
    assert payload["epoch"] == 3
    assert payload["best_metric"] == 1.23
    m2 = chg_net_from_config(default_cfg)
    m2.load_state_dict(payload["model_state"])
    assert torch.allclose(
        m.class_embed.weight,
        m2.class_embed.weight,
    )
