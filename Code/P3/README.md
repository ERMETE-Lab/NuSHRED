# P3: SHRED on the DYNASTY Experimental Facility

Notebooks supporting:

> Riva, S., Missaglia, A., Introini, C., Kutz, J. N. & Cammi, A. (2026). From Models To Experiments: Shallow Recurrent Decoder Networks on the DYNASTY Experimental Facility. [arXiv:2503.08907](https://arxiv.org/abs/2503.08907)

## Dataset

**[DYNASTY](https://doi.org/10.5281/zenodo.20554287)** — RELAP5 model and experimental data (single transient and parametric cases).

```bash
uv run python Code/download_datasets.py --files DYNASTY
```

Data path in notebooks: `$NUSHRED_DATA_DIR/DYNASTY/`.

Simulation data shared on Zenodo are rescaled to $[0, 1]$ for open distribution.

## Requirements

```bash
uv sync
```

## Notebooks

| Notebook | Description |
| -------- | ----------- |
| `01_shred_verification_parametric.ipynb` | Verification on synthetic (model-only) parametric transients |
| `02a_shred_validation_parametric.ipynb` | Validation — train on simulation, test on experiment |
| `02b_shred_validation_forecasting.ipynb` | Forecasting beyond the training window |
