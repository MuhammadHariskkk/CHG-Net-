# Data directory

## Layout (recommended)

```
data/
├── README.md           # this file
├── label_mapping.md    # human-readable taxonomy → see also YAML `labels.*`
├── raw/sdd/            # untouched SDD downloads (gitignored)
└── processed/sdd/    # preprocessed tensors/records (gitignored)
```

## Stanford Drone Dataset

1. Obtain SDD annotations and videos per the dataset license.
2. Place raw XML/csv (or official layout) under `data/raw/sdd/` — exact expected layout is documented in **Phase 2** (`preprocess_data.py`).

## Phase 1 note

Preprocessing scripts and dataset classes are not included yet; this folder documents intent and `label_mapping` policy.
