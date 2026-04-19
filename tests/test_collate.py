from __future__ import annotations

from typing import Any

from chgnet.datasets.collate import collate_sdd_batch

from tests.helpers import make_preprocessed_sample


def test_collate_pads_to_n_max(default_cfg: dict[str, Any]) -> None:
    data = default_cfg["data"]
    o, p = int(data["obs_len"]), int(data["pred_len"])
    s1 = make_preprocessed_sample(obs_len=o, pred_len=p, n_agents=2)
    s2 = make_preprocessed_sample(obs_len=o, pred_len=p, n_agents=4)
    batch = collate_sdd_batch([s1, s2])
    B, T, n_max, _ = batch["obs_xy"].shape
    assert B == 2
    assert T == o
    assert n_max == 4
    assert batch["fut_xy"].shape == (B, p, n_max, 2)
    assert batch["agent_id"][0, 2] == -1
