"""Synthetic SDD-style samples for unit tests."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_preprocessed_sample(
    *,
    obs_len: int,
    pred_len: int,
    n_agents: int,
    scene_id: str = "test_scene",
    sequence_id: str = "test_seq",
) -> dict[str, Any]:
    """Minimal Phase-2-style sample dict (NumPy) for collate tests."""
    need = obs_len + pred_len
    slot = np.arange(need, dtype=np.int32)
    obs_xy = np.random.randn(obs_len, n_agents, 2).astype(np.float32)
    fut_xy = np.random.randn(pred_len, n_agents, 2).astype(np.float32)
    obs_vel = np.zeros((obs_len, n_agents, 2), dtype=np.float32)
    obs_heading = np.zeros((obs_len, n_agents), dtype=np.float32)
    obs_rel = np.zeros((obs_len, n_agents, 2), dtype=np.float32)
    obs_valid = np.ones((obs_len, n_agents), dtype=bool)
    fut_valid = np.ones((pred_len, n_agents), dtype=bool)
    agent_id = np.arange(n_agents, dtype=np.int64)
    # Full taxonomy index order: cyclist=0, other=1, pedestrian=2, ... (see label_mapping._FULL_VOCAB_ORDER).
    ped_idx = 2
    class_idx = np.full((n_agents,), ped_idx, dtype=np.int64)
    raw_label = ["pedestrian"] * n_agents
    mapped_label = ["pedestrian"] * n_agents
    return {
        "scene_id": scene_id,
        "sequence_id": sequence_id,
        "resampled_frame_min": 0,
        "window_k_start": 0,
        "window_k_end_obs": obs_len - 1,
        "window_k_end_fut": need - 1,
        "slot_indices": slot,
        "obs_xy": obs_xy,
        "fut_xy": fut_xy,
        "obs_vel": obs_vel,
        "obs_heading": obs_heading,
        "obs_rel_disp": obs_rel,
        "obs_valid": obs_valid,
        "fut_valid": fut_valid,
        "agent_id": agent_id,
        "class_idx": class_idx,
        "raw_label": raw_label,
        "mapped_label": mapped_label,
    }
