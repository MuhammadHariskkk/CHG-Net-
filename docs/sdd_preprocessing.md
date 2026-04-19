# SDD preprocessing (Phase 2)

## paper-specified

- Default **8** observation and **12** future steps at **2.5 Hz** (`configs/default.yaml` → `data.obs_len`, `data.pred_len`, `data.sample_rate_hz`).

## Raw annotations (**engineering assumption**)

- Expected file: `<raw_root>/<scene>/<sequence>/annotations.txt` (or `<scene>/annotations.txt` with `sequence_id` `_root`).
- Line format (Stanford Drone Dataset convention):  
  `track_id xmin ymin xmax ymax frame lost occluded generated label…`  
  `label` may contain spaces; this implementation joins all tokens after the ninth field and strips quotes.
- Bounding-box centers: \(x = (x_{\min}+x_{\max})/2\), \(y = (y_{\min}+y_{\max})/2\) in **image-plane pixels**.

## Resampling

- Native grid uses `data.source_fps` (default **30** Hz; **not** stated in CHG-Net paper—override if your videos differ).
- Target index \(k\) maps to native frame `frame_min + k * (source_fps / sample_rate_hz)` with **nearest-neighbor** lookup per track (**engineering assumption**).

## Windows & agents

- Sliding windows stride `data.window_stride` (default **1**) on the resampled timeline.
- If `data.require_full_window_valid` is **true** (default), an agent appears in a window only if `lost==0`, coordinates are in optional `data.image_bounds`, and the agent is valid on **every** observation and future step.
- If **false**, any agent valid at the last observation step may be included; `obs_valid` / `fut_valid` encode per-step visibility for graph masking later.
- Agent tensor order: **sorted ascending by `track_id`**.

## Heading, velocity, relative displacement

- Velocity on observed steps: backward finite differences divided by \(1/\text{sample\_rate\_hz}\); replicate first step from the second (**engineering assumption**).
- Heading: `atan2(vy, vx)`; if speed \(< \varepsilon\) (`1e-3` px/s scale) or invalid, reuse **last valid** heading, else **0**.
- `obs_rel_disp`: position at each observation step minus position at the **last observed** step (NaN if invalid).

## Processed artifacts

Written under `data.processed_root`:

| File | Content |
|------|---------|
| `preprocess_meta.json` | Vocabulary, taxonomy mode, split snapshot, global/per-scene label counts, raw→mapped edge counts, dropped stats, config snapshot |
| `dataset_index.json` | Chunk list, `sample_scene_ids`, `sample_sequence_ids`, `total_samples` |
| `samples_chunk_*.pt` | `{"samples": [ {dict}, ... ]}` each dict is one window |

## Single-sample dict schema (NumPy inside `.pt`)

| Key | Shape / type |
|-----|----------------|
| `scene_id`, `sequence_id` | `str` |
| `resampled_frame_min` | `int` native frame anchor |
| `window_k_start`, `window_k_end_obs`, `window_k_end_fut` | `int` indices on resampled grid |
| `slot_indices` | `(obs_len+pred_len,) int32` indices on the **resampled** timeline (k), not raw SDD frame IDs; CARLA CSV `frame_index` / `time_index` in Phase 9 must derive from these slots plus `resampled_frame_min`, `data.sample_rate_hz`, and `data.source_fps` (see `docs/carla_export.md`). |
| `obs_xy`, `obs_vel`, `obs_rel_disp` | `(obs_len, N, 2) float32` |
| `obs_heading` | `(obs_len, N) float32` |
| `fut_xy` | `(pred_len, N, 2) float32` |
| `obs_valid`, `fut_valid` | `(obs_len, N)` / `(pred_len, N) bool` |
| `agent_id` | `(N,) int64` |
| `raw_label`, `mapped_label` | `list[str]` length `N` |
| `class_idx` | `(N,) int64` model class indices |

**Per-frame agent collections:** rows `obs_valid[t, :]` with `obs_xy[t, :, :]` (and velocity/heading) are the active agents at observation time `t` for graph construction.

## Batched schema (`collate_sdd_batch`)

Leading batch dimension `B`; `N_max` pads agents. See `chgnet/datasets/collate.py` for exact tensor shapes and `num_agents` per row.

## Splits

- `data.splits` maps `train` / `val` / `test` → scene folder names.
- Defaults are **engineering assumptions**; override in YAML to match your benchmark.
- If `data.splits` is missing or all lists are empty, `SDDProcessedDataset` loads **all** samples (**backward compatibility**).

## `SDDProcessedDataset` example

```python
from chgnet.utils.config import load_config
from chgnet.datasets import SDDProcessedDataset
from torch.utils.data import DataLoader
from chgnet.datasets.collate import collate_sdd_batch

cfg = load_config("configs/little.yaml", defaults_first="configs/default.yaml")
ds = SDDProcessedDataset(cfg["data"]["processed_root"], cfg, split="train")
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_sdd_batch)
batch = next(iter(loader))
```

## Preprocessing command

```bash
cd chg_net
python scripts/preprocess_data.py --config configs/sdd.yaml --defaults configs/default.yaml
```

Optional overrides: `--raw-root`, `--processed-root`, `--override data.scene_filter=[coupa]`.

## Label inspection

```bash
python scripts/inspect_labels.py --config configs/sdd.yaml --defaults configs/default.yaml
```
