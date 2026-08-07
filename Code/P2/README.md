# P2: Parametric State Estimation in Circulating Fuel Reactors

Notebooks supporting:

> Riva, S., Introini, C., Kutz, J. N. & Cammi, A. (2025). Towards Efficient Parametric State Estimation in Circulating Fuel Reactors with Shallow Recurrent Decoder Networks. [arXiv:2503.08904](https://arxiv.org/abs/2503.08904)

## Dataset

**[MSFR](https://doi.org/10.5281/zenodo.20554287)** — parametric ULOFF transients (same archive as P1).

```bash
uv run python Code/download_datasets.py --files MSFR
```

Data path in notebooks: `$NUSHRED_DATA_DIR/MSFR/`.

## Requirements

```bash
uv sync
```

## Notebooks

| Notebook | Description |
| -------- | ----------- |
| `01_svd.ipynb` | SVD analysis of the parametric MSFR dataset |
| `02_shred_outcore.ipynb` | **Main paper notebook** — out-core fast-flux sensors |
| `03_sensitivity_ensemblesize.ipynb` | Sensitivity to ensemble size |
| `04a_shred_mobile_sensors.ipynb` | Mobile in-core sensors (precursor group 1) |
| `04b_shred_mobile_probes.ipynb` | Mobile probes (position only) |
| `05_comparison_shred_online.ipynb` | Comparison of sensing strategies |
