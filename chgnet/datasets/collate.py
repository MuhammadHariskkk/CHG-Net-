"""Batch collation for variable agent counts (CHG graph–ready schema).

**engineering assumption**: pad agent dimension with NaN for coordinates, False for masks,
``-1`` for ``agent_id`` and ``class_idx``, empty string for labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def collate_sdd_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of dataset samples into a batched dict of torch tensors / lists.

    Shapes:
        ``obs_xy``: ``(B, obs_len, N_max, 2)``
        ``fut_xy``: ``(B, pred_len, N_max, 2)``
        ``obs_vel``: ``(B, obs_len, N_max, 2)``
        ``obs_heading``: ``(B, obs_len, N_max)``
        ``obs_rel_disp``: ``(B, obs_len, N_max, 2)``
        ``obs_valid``, ``fut_valid``: ``(B, T, N_max)``
        ``agent_id``: ``(B, N_max)`` int64, padded ``-1``
        ``class_idx``: ``(B, N_max)`` int64, padded ``-1``
        ``raw_label``, ``mapped_label``: length-``B`` lists of length-``N_max`` lists (strings).
    """
    if not samples:
        raise ValueError("Empty batch")
    b = len(samples)
    obs_len = samples[0]["obs_xy"].shape[0]
    pred_len = samples[0]["fut_xy"].shape[0]
    n_max = max(
        max(int(s["obs_xy"].shape[1]), int(s["fut_xy"].shape[1])) for s in samples
    )

    def pad_agent_array(x: np.ndarray, fill: float | bool) -> np.ndarray:
        # x shape (T, n, C) or (T, n)
        t = x.shape[0]
        if x.ndim == 2:
            out = np.full((t, n_max), fill, dtype=x.dtype)
            n = x.shape[1]
            out[:, :n] = x
            return out
        c = x.shape[2]
        out = np.full((t, n_max, c), fill, dtype=x.dtype)
        n = x.shape[1]
        out[:, :n, :] = x
        return out

    obs_xy = np.stack(
        [pad_agent_array(s["obs_xy"], np.nan) for s in samples],
        axis=0,
    )
    fut_xy = np.stack(
        [pad_agent_array(s["fut_xy"], np.nan) for s in samples],
        axis=0,
    )
    obs_vel = np.stack(
        [pad_agent_array(s["obs_vel"], 0.0) for s in samples],
        axis=0,
    )
    obs_heading = np.stack(
        [pad_agent_array(s["obs_heading"], 0.0) for s in samples],
        axis=0,
    )
    obs_rel_disp = np.stack(
        [pad_agent_array(s["obs_rel_disp"], np.nan) for s in samples],
        axis=0,
    )
    obs_valid = np.stack(
        [pad_agent_array(s["obs_valid"], False) for s in samples],
        axis=0,
    )
    fut_valid = np.stack(
        [pad_agent_array(s["fut_valid"], False) for s in samples],
        axis=0,
    )

    agent_id = np.full((b, n_max), -1, dtype=np.int64)
    class_idx = np.full((b, n_max), -1, dtype=np.int64)
    raw_label_batch: list[list[str]] = []
    mapped_label_batch: list[list[str]] = []

    for i, s in enumerate(samples):
        n = int(s["agent_id"].shape[0])
        agent_id[i, :n] = s["agent_id"]
        class_idx[i, :n] = s["class_idx"]
        rl = list(s["raw_label"])
        ml = list(s["mapped_label"])
        while len(rl) < n_max:
            rl.append("")
            ml.append("")
        raw_label_batch.append(rl)
        mapped_label_batch.append(ml)

    slot_stack = np.stack([s["slot_indices"] for s in samples], axis=0)

    batch: dict[str, Any] = {
        "scene_id": [str(s["scene_id"]) for s in samples],
        "sequence_id": [str(s["sequence_id"]) for s in samples],
        "resampled_frame_min": torch.tensor([int(s["resampled_frame_min"]) for s in samples], dtype=torch.long),
        "window_k_start": torch.tensor([int(s["window_k_start"]) for s in samples], dtype=torch.long),
        "window_k_end_obs": torch.tensor([int(s["window_k_end_obs"]) for s in samples], dtype=torch.long),
        "window_k_end_fut": torch.tensor([int(s["window_k_end_fut"]) for s in samples], dtype=torch.long),
        "slot_indices": torch.from_numpy(slot_stack.astype(np.int64)),
        "obs_xy": torch.from_numpy(obs_xy.astype(np.float32)),
        "fut_xy": torch.from_numpy(fut_xy.astype(np.float32)),
        "obs_vel": torch.from_numpy(obs_vel.astype(np.float32)),
        "obs_heading": torch.from_numpy(obs_heading.astype(np.float32)),
        "obs_rel_disp": torch.from_numpy(obs_rel_disp.astype(np.float32)),
        "obs_valid": torch.from_numpy(obs_valid),
        "fut_valid": torch.from_numpy(fut_valid),
        "agent_id": torch.from_numpy(agent_id),
        "class_idx": torch.from_numpy(class_idx),
        "raw_label": raw_label_batch,
        "mapped_label": mapped_label_batch,
        "num_agents": torch.tensor([int(s["agent_id"].shape[0]) for s in samples], dtype=torch.long),
    }
    return batch
