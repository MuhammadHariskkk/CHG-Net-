from __future__ import annotations

from typing import Any

import torch

from chgnet.exporters.carla_csv import build_export_rows, sort_export_rows
from chgnet.losses import gmm_trajectory_nll
from chgnet.metrics import ade_fde_deterministic
from chgnet.models.chg_net import chg_net_from_config


def test_chgnet_forward_and_loss(default_cfg: dict[str, Any], synthetic_batch: dict[str, Any]) -> None:
    model = chg_net_from_config(default_cfg)
    out = model(synthetic_batch)
    assert out.gmm.means.shape[0] == synthetic_batch["obs_xy"].shape[0]
    loss = gmm_trajectory_nll(
        out.gmm,
        synthetic_batch["fut_xy"],
        node_mask=out.node_mask,
        fut_valid=synthetic_batch["fut_valid"],
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    ade, fde = ade_fde_deterministic(
        out.gmm.deterministic_trajectory,
        synthetic_batch["fut_xy"],
        node_mask=out.node_mask,
        fut_valid=synthetic_batch["fut_valid"],
    )
    assert torch.isfinite(ade) and torch.isfinite(fde)


def test_build_export_rows_schema(default_cfg: dict[str, Any], synthetic_batch: dict[str, Any]) -> None:
    model = chg_net_from_config(default_cfg)
    out = model(synthetic_batch)
    rows = build_export_rows(synthetic_batch, out, default_cfg, export_run_id="pytest")
    assert len(rows) > 0
    assert rows[0]["agent_class"] == rows[0]["mapped_label"]
    sorted_rows = sort_export_rows(rows)
    assert len(sorted_rows) == len(rows)
