"""Load preprocessed SDD windows written by :func:`chgnet.datasets.preprocessing.run_preprocessing`."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from chgnet.datasets.scene_split import scenes_for_split


def _torch_load_compat(path: Path, map_location: str | None = None) -> Any:
    kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


class SDDProcessedDataset(Dataset):
    """Index-addressable dataset over chunked ``.pt`` sample lists.

    Args:
        processed_root: Directory containing ``dataset_index.json`` and chunk files.
        cfg: Merged config (uses ``data.preprocess`` paths and ``data.splits``).
        split: ``train`` | ``val`` | ``test`` — filters samples by ``scene_id`` membership.
    """

    def __init__(
        self,
        processed_root: str | Path,
        cfg: Mapping[str, Any],
        split: str = "train",
    ) -> None:
        self.root = Path(processed_root)
        self.cfg = cfg
        self.split = split
        data = cfg["data"]
        pre = data.get("preprocess", {})
        index_name = pre.get("index_filename", "dataset_index.json")
        index_path = self.root / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"Dataset index not found: {index_path.resolve()}")
        self._index = json.loads(index_path.read_text(encoding="utf-8"))
        self._chunks_meta = list(self._index["chunks"])
        self._cum: list[int] = []
        total = 0
        for ch in self._chunks_meta:
            total += int(ch["num_samples"])
            self._cum.append(total)
        self._total_samples = int(self._index.get("total_samples", total))

        scene_ids = self._index.get("sample_scene_ids")
        if scene_ids is None:
            raise ValueError(
                "dataset_index.json missing sample_scene_ids; re-run preprocessing with Phase 2 code."
            )
        allowed = scenes_for_split(split, cfg)
        if allowed is None:
            self._filtered_indices = list(range(self._total_samples))
        else:
            self._filtered_indices = [i for i, sid in enumerate(scene_ids) if sid in allowed]
        if not self._filtered_indices:
            raise ValueError(
                f"No samples for split={split!r} with scenes {sorted(allowed or [])!r} under {self.root}."
            )

        self._cache: dict[str, list[dict[str, Any]]] = {}

    def __len__(self) -> int:
        return len(self._filtered_indices)

    def _load_chunk(self, chunk_path: str) -> list[dict[str, Any]]:
        if chunk_path in self._cache:
            return self._cache[chunk_path]
        payload = _torch_load_compat(self.root / chunk_path, map_location="cpu")
        samples = payload["samples"]
        self._cache[chunk_path] = samples
        return samples

    def _global_to_local(self, global_idx: int) -> tuple[str, int]:
        lo = 0
        for ci, hi in enumerate(self._cum):
            if global_idx < hi:
                chunk = self._chunks_meta[ci]
                offset = global_idx - lo
                return str(chunk["path"]), int(offset)
            lo = hi
        raise IndexError(f"global_idx {global_idx} out of range")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self._filtered_indices):
            raise IndexError(idx)
        g_idx = self._filtered_indices[idx]
        cpath, offset = self._global_to_local(g_idx)
        chunk = self._load_chunk(cpath)
        return chunk[offset]
