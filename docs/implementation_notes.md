# Implementation notes: CHG-Net

This document grows each phase. Entries are tagged:

- **paper-specified**: explicitly stated in the paper (methodology / cited hyperparameters).
- **engineering assumption**: reasonable choice where the paper is silent or under-specified.
- **recommended extension**: optional improvement, not required for baseline replication.

## Phase 1 (scaffold)

### paper-specified

- Observation length 8, prediction length 12; SDD sampling 2.5 Hz.
- Learning rate 1e-3, batch size 64 (training defaults in `configs/default.yaml`).
- Class embedding dimension 16, graph hidden dimension 64.
- Micro-TCN: kernel size 3, two layers; GMM decoder with *K* = 3 components.
- Interaction mask combines distance, motion-direction, and heading terms with weights *α, β, γ* (exact formulas implemented in **Phase 4**).

### engineering assumption

- Optimizer defaults to Adam; weight decay 0 unless ablation requires otherwise.
- `max_epochs`, `log_every_n_steps`, and checkpoint/history paths are **not** paper-specified; they are run-scaffold defaults.
- `min_track_len_frames: 20` matches `obs_len + pred_len` for a minimal valid window (20 = 8+12); may be tightened per dataset cleanup.
- CARLA CSV `file_layout`, row-sorting keys, and optional mask thresholding are configurable for reproducibility and downstream tooling.
- `configs/sdd.yaml` does not fix train/val/test lists in Phase 1; **Phase 2** (`scene_split.py`) will own canonical splits and document them.

### recommended extension

- Mixed precision training, EMA weights, and Weights & Biases logging.

---

## Phase 2 (SDD preprocessing)

### paper-specified

- Window lengths and output sampling rate align with `configs/default.yaml` (`obs_len`, `pred_len`, `sample_rate_hz`).

### engineering assumption

- Native SDD `annotations.txt` column order and 30 Hz default (`data.source_fps`).
- Nearest-neighbor resampling to 2.5 Hz; sliding-window stride and strict all-step validity defaults.
- Default `data.splits` scene lists are benchmark scaffolding only (override in YAML).
- `motorcycle` raw alias maps to `cyclist` (sparse in SDD; adjust aliases if your release differs).

### recommended extension

- Linear interpolation between native frames instead of nearest-neighbor.
- XML / alternate SDD mirrors via a pluggable reader.

---

## Phase 3 (CHG features + adjacency)

### paper-specified

- Node inputs include final observed position, velocity, heading, and discrete class identity.
- Edge inputs include relative displacement, relative velocity, cos(θ_ij), cos(φ_ij), and endpoint class linkage.

### engineering assumption

- Exact trigonometric definitions for θ_ij and φ_ij on directed edges *i* → *j* are documented in
  ``chgnet/graph/features.py`` (displacement vs source velocity; heading unit-dot-product).
- Directed clique without self-loops; edge ordering is row-major over (*i*, *j*) with *i* ≠ *j*.

### recommended extension

- Extra occlusion or FOV gating before masking (not in baseline CHG-Net description).

---

## Phase 4 (direction-aware mask)

### paper-specified

- w_d = exp(−‖**p_i** − **p_j**‖), w_v = (1 + cos θ_ij)/2, w_h = (1 + cos φ_ij)/2, m_ij = α w_d + β w_v + γ w_h,
  **ẽ**_ij = m_ij **e**_ij (see ``chgnet/graph/mask.py``).

### engineering assumption

- ‖**p_i** − **p_j**‖ from ``edge_attr`` Δp (same as Phase 3); optional ``masking.apply_threshold`` / ``threshold``;
  edges touching ``node_valid == False`` are zeroed for padded batches.

### recommended extension

- Normalize m_ij (e.g. softmax over neighbors) if attention stability requires it in ablations.

---

## Phase 5 (ST-GCN block)

### paper-specified

- Spatial-temporal processing on the CHG: neighbor aggregation driven by edge features, temporal
  filtering along the observation sequence.

### engineering assumption

- Incoming-neighbor softmax attention on Φ(**ẽ**); message MLP on [**h**_i ‖ Φ(**ẽ**)]; self linear
  on **h**_j; Conv1d temporal kernel default 3 with same-length padding; 2D ``edge_attr`` broadcast
  over time unless a 4D tensor is supplied.

### recommended extension

- Multi-layer ST-GCN stack or deeper temporal stacks; rebuild **ẽ** per timestep from ``obs_xy``.

---

## Phase 6 (Micro-TCN + GMM decoder)

### paper-specified

- Micro-TCN: kernel 3, two layers, dilated Conv1d, residual refinement; GMM with *K* = 3 modes;
  deterministic pick = argmax mixture weight.

### engineering assumption

- Micro-TCN I/O layout ``(B, C, T)``; dilations ``dilation_base**layer``; residual ``ReLU(x + conv(x))`` per block.
- GMM heads from a two-layer MLP; ``σ = softplus(raw) + 1e-4``; decoder ``hidden_dim = max(64, 4*D, graph_hidden_dim)``.
- Mean outputs live in the same space as training targets (defined when wiring loss in Phase 8).

### recommended extension

- Predict residuals relative to last observed position for numerical stability.

---

*Subsequent phases append full CHG-Net integration, training loop, and export fidelity notes here.*
