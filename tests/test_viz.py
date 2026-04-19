from __future__ import annotations

from pathlib import Path
from typing import Any

from chgnet.models.chg_net import chg_net_from_config
from chgnet.viz.trajectory_plot import plot_batch_item_trajectories


def test_plot_batch_item_writes_png(
    default_cfg: dict[str, Any], synthetic_batch: dict[str, Any], tmp_path: Path
) -> None:
    model = chg_net_from_config(default_cfg)
    out = model(synthetic_batch)
    path = tmp_path / "t.png"
    plot_batch_item_trajectories(synthetic_batch, default_cfg, 0, out, out_path=path)
    assert path.is_file()
    assert path.stat().st_size > 1000
