# Code

Application notebooks and utilities for the NuSHRED papers (P1–P5). Each `P*` folder is self-contained and maps to one publication — see the [main README](../README.md) for installation, datasets, and citations.

## Layout

| Folder | Paper | Dataset(s) | README |
| ------ | ----- | ---------- | ------ |
| [P1/](P1/) | Robust out-core state estimation (MSFR) | MSFR | [P1/README.md](P1/README.md) |
| [P2/](P2/) | Parametric state estimation (MSFR) | MSFR | [P2/README.md](P2/README.md) |
| [P3/](P3/) | V&V on DYNASTY facility | DYNASTY | [P3/README.md](P3/README.md) |
| [P4/](P4/) | Constrained sensing (TRIGA) | TRIGA | [P4/README.md](P4/README.md) |
| [P5/](P5/) | Multi-fidelity SHRED | MSFR, LRA-neutronics, RDA | [P5/README.md](P5/README.md) |

Shared helpers (`tools.py`, `plots.py`, `scalers.py`) live inside each paper folder when needed.

## Dataset utilities

### Download from Zenodo

```bash
# all datasets
uv run python Code/download_datasets.py

# specific datasets
uv run python Code/download_datasets.py --files MSFR DYNASTY

# legacy names from the first Zenodo release (D1/D2 → MSFR, D3 → DYNASTY, …)
uv run python Code/download_datasets.py --files D1 D3
```

Uses the [concept DOI](https://doi.org/10.5281/zenodo.13789584) (record `13789584`), which always resolves to the latest version. Legacy archive names (`D1.zip`, …) are detected automatically when the renamed zips are not yet on Zenodo.

By default, archives are extracted under `NuSHRED_Datasets/` at the repo root. Override the location with the `NUSHRED_DATA_DIR` environment variable (see [`.env.example`](../.env.example)).

### Prepare for upload

```bash
uv run python Code/prepare_datasets4upload.py
```

Cleans cache/junk files, zips each dataset subfolder, and writes `checksums.txt`. Run this before publishing a new Zenodo version.

## Data paths in notebooks

Notebooks resolve data via:

```python
data_root = os.environ.get("NUSHRED_DATA_DIR", "../../NuSHRED_Datasets")
```

When running from a `Code/P*/` notebook, the default relative path points to `NuSHRED_Datasets/` at the repo root. Set `NUSHRED_DATA_DIR` if your data live elsewhere.
