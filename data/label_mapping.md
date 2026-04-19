# SDD label mapping (policy)

Configurable via `configs/default.yaml` → `labels` and implemented in **Phase 2** (`chgnet/datasets/label_mapping.py`).

## Internal taxonomy (full)

| Internal class | Typical raw SDD strings (non-exhaustive) |
|----------------|------------------------------------------|
| `pedestrian`   | `pedestrian`, `person`, `Pedestrian`, … |
| `cyclist`      | `biker`, `Biker`, `cyclist`, `bike`, … |
| `vehicle`      | `car`, `Car`, `cart`, `bus`, `vehicle`, … |
| `skater`       | `skater`, `Skater`, `skateboard`, … |
| `other`        | unknown / empty / misc |

*(Table is illustrative; Phase 2 code normalizes case and aliases deterministically.)*

## Collapsed taxonomy (`labels.taxonomy_mode: collapsed_3`)

- Target vocabulary is exactly three internal classes: `pedestrian`, `cyclist`, `vehicle`.
- **`skater` handling**: controlled by `labels.collapse_skater_as` (`cyclist` default in YAML — **engineering assumption**; override to map skater elsewhere if desired).
- Raw unknowns map to internal `other` under `labels.unknown_internal_class`, then **Phase 2** must merge `other` into one of `{pedestrian, cyclist, vehicle}` for training/export (default policy to be implemented and documented in `label_mapping.py`; e.g. map `other` → `pedestrian` if no finer signal). Until Phase 2 ships, treat this as an explicit open policy to avoid silent mismatch between metrics (3-class head) and CSV `mapped_label`.

## Deterministic vocabulary

- **Full** mode (`labels.taxonomy_mode: full`): fixed lexicographic order  
  `cyclist` (0), `other` (1), `pedestrian` (2), `skater` (3), `vehicle` (4) — see `chgnet/datasets/label_mapping.py` (`_FULL_VOCAB_ORDER`).
- **Collapsed** mode (`collapsed_3`): `cyclist` (0), `pedestrian` (1), `vehicle` (2) — `_COLLAPSED_VOCAB_ORDER`.
- The same order is written to `preprocess_meta.json` as `class_vocabulary` / `class_to_index`.

## paper-specified vs assumption

- **paper-specified**: multi-class prediction on SDD with heterogeneous agents.
- **engineering assumption**: exact string alias list and collapse defaults; YAML makes them explicit without silent redefinition.
