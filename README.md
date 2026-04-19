# CHG-Net 

PyTorch implementation of **CHG-Net: A Unified Direction-Aware Graph Framework for Multi-Class Trajectory Prediction** (Stanford Drone Dataset–oriented, **CARLA-first CSV export**, Matplotlib debug plots).

The repo includes preprocessing, the full graph + ST-GCN + Micro-TCN + GMM model, training, export, visualization, and **pytest** smoke tests.

## Setup

Requires **Python 3.10+**.

```bash
cd chg_net
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
# Optional: same as requirements dev line
pip install -e ".[dev]"
```

## Preprocess SDD

```bash
python scripts/preprocess_data.py --config configs/sdd.yaml --defaults configs/default.yaml
python scripts/inspect_labels.py --config configs/sdd.yaml --defaults configs/default.yaml
```

See [`docs/sdd_preprocessing.md`](docs/sdd_preprocessing.md) for annotation layout, tensor shapes, and split behavior.

## Train

```bash
python scripts/train.py --config configs/default.yaml
# Resume:
python scripts/train.py --config configs/default.yaml --resume outputs/checkpoints/checkpoint_best.pt
```

## CARLA CSV export

```bash
python scripts/export_carla_csv.py --config configs/default.yaml --checkpoint outputs/checkpoints/checkpoint_best.pt --split test
```

Specification: [`docs/carla_export.md`](docs/carla_export.md).

## Debug trajectory plots (Matplotlib)

```bash
python scripts/plot_trajectories.py --config configs/default.yaml --checkpoint outputs/checkpoints/checkpoint_best.pt --split val
```

## Tests (Phase 11)

From the `chg_net` directory, with the package installed editable:

```bash
python -m pytest tests -q
```

`pytest.ini` configures the `tests/` directory. Synthetic batches **do not** require SDD data on disk.

## Configuration

- Global defaults: [`configs/default.yaml`](configs/default.yaml) (8 observed / 12 predicted steps, 2.5 Hz, batch 64, LR 1e-3, class embedding 16, graph hidden 64, micro-TCN kernel 3 × 2 layers, GMM *K* = 3, masking *α, β, γ*).
- Scene snippets (e.g. [`configs/little.yaml`](configs/little.yaml)) override `data.scene_filter`.
- Merge order: pass multiple YAML paths to `load_config` (later wins); dotted overrides are supported in [`chgnet/utils/config.py`](chgnet/utils/config.py).

## Stanford Drone Dataset (SDD)

- Annotations use string labels; the pipeline maps raw strings to an internal taxonomy via `labels.taxonomy_mode` (`full` vs `collapsed_3`). See [`data/label_mapping.md`](data/label_mapping.md) and [`docs/sdd_preprocessing.md`](docs/sdd_preprocessing.md).
- Trajectories are resampled to **2.5 Hz** with **8** observation and **12** prediction frames (paper-aligned defaults in YAML).
- Coordinates default to **image-plane** unless a world transform is configured for export (`carla_export`).

## Repository layout

```
chg_net/
├── README.md
├── requirements.txt
├── setup.py
├── pytest.ini
├── configs/
├── data/
├── docs/
├── scripts/
├── chgnet/           # package
├── tests/            # Phase 11 pytest
└── outputs/          # gitignored by default
```

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/implementation_notes.md`](docs/implementation_notes.md) | Paper choices |
| [`docs/architecture.md`](docs/architecture.md) | Module map and data flow |
| [`docs/sdd_preprocessing.md`](docs/sdd_preprocessing.md) | SDD robustness and tensors |
| [`docs/carla_export.md`](docs/carla_export.md) | CSV schema and coordinate policy |

## License

MIT (recommended; confirm with your institution before publishing).
