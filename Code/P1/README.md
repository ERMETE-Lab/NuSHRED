# P1: Robust State Estimation from Partial Out-Core Measurements

Notebooks supporting:

> Riva, S., Introini, C., Cammi, A., & Kutz, J. N. (2025). Robust State Estimation from Partial Out-Core Measurements with Shallow Recurrent Decoder for Nuclear Reactors. *Progress in Nuclear Energy*, vol. 189, pp. 105928. [doi:10.1016/j.pnucene.2025.105928](https://doi.org/10.1016/j.pnucene.2025.105928)

## Dataset

**[MSFR](https://doi.org/10.5281/zenodo.20554287)** — single-transient ULOFF reconstruction case (included in the MSFR archive).

```bash
uv run python Code/download_datasets.py --files MSFR
```

Data path in notebooks: `$NUSHRED_DATA_DIR/MSFR/` (default: `NuSHRED_Datasets/MSFR/`).

## Requirements

Base install plus the P1 optional dependency (pyforce for EIM/GEIM notebooks):

```bash
uv sync --extra p1
```

## Notebooks

| Notebook | Description |
| -------- | ----------- |
| `01a_svd.ipynb` | SVD analysis of the MSFR dataset |
| `01b_compute_measures.ipynb` | Compute out-core sensor measures |
| `02a_shred_uq.ipynb` | **Main paper notebook** — ensemble SHRED with uncertainty quantification |
| `02b_PDEresidual_check.ipynb` | PDE residual and mass-conservation checks |
| `02c_eim.ipynb` | EIM sensor placement (requires pyforce) |
| `02d_geim.ipynb` | GEIM sensor placement (requires pyforce) |
