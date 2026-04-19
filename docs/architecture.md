# Architecture (target)

**Phase 1** defines packaging and configuration only. The following describes the planned module graph for CHG-Net.

## Planned data flow

1. **SDD preprocessing** (`chgnet/datasets/preprocessing.py`, **Phase 2**): raw tracks → windows with positions, velocities, headings, raw/mapped labels, masks.
2. **Collate** (`chgnet/datasets/collate.py`): variable-agent batches preserving scene/frame/agent metadata.
3. **CHG graph** (`chgnet/graph/`): **Phase 3** builds node tensors, directed clique `edge_index`, edge attrs (Δ*p*, Δ*v*, cos *θ*, cos *φ*) + endpoint class indices. **Phase 4** (`mask.py`) applies *m_ij* to edge features. **Phase 5** (`stgcn_block.py`) runs spatial attention + temporal Conv1d on ``(B, T, N, F)`` with masked ``edge_attr``.
4. **ST-GCN** (**Phase 5**): spatial message passing over masked edges + temporal convolution along observation steps.
5. **Micro-TCN** (`chgnet/models/micro_tcn.py`, **Phase 6**): dilated 1D conv + residuals on ``(B, C, T)``.
6. **GMM decoder** (`chgnet/models/gmm_decoder.py`, **Phase 6**): *K* modes, diagonal ``σ``; :class:`GMMDecoderOutput` exposes probs, deterministic trajectory, all modes (**Phase 7** wires into `chg_net.py`).
7. **Loss / metrics** (**Phase 8**): GMM NLL; ADE/FDE on deterministic trajectory.
8. **CARLA export** (**Phase 9**): `chgnet/exporters/carla_csv.py` builds sorted CSV rows (`build_export_rows`, etc.).

## Package layout

| Path | Responsibility |
|------|----------------|
| `chgnet/utils/` | Config, seed, log, checkpoint, later geometry/viz |
| `chgnet/datasets/` | SDD I/O, labels, splits, collate |
| `chgnet/graph/` | Features, adjacency, mask, ST-GCN, builder |
| `chgnet/models/` | micro-TCN, GMM, full CHG-Net |
| `chgnet/losses/` | GMM NLL |
| `chgnet/metrics/` | ADE, FDE, deterministic helpers |
| `chgnet/trainers/` | Train/val loops, history |
| `chgnet/exporters/` | CARLA CSV schema + writers |
| `chgnet/viz/` | Matplotlib debug trajectory plots (Phase 10) |
| `tests/` | Pytest smoke tests (Phase 11) |

## Integration risks (preview)

- Batch schema must stay stable from **Phase 2** through graph builder and exporter (**Phase 9**).
- Class vocabulary order must be deterministic and saved in preprocessing metadata for embedding consistency.
