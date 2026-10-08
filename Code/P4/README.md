# P4: Constrained Sensing on a TRIGA Mark II Reactor

Notebooks supporting:

> Riva, S., Introini, C., Kutz, J. N. & Cammi, A. (2027). Constrained Sensing and Reliable State Estimation with Shallow Recurrent Decoders on a TRIGA Mark II Reactor. *Chemical Engineering Science*, 338, 125037. [doi:10.1016/j.ces.2026.125037](https://doi.org/10.1016/j.ces.2026.125037) ([arXiv:2510.12368](https://arxiv.org/abs/2510.12368))

## Dataset

**[TRIGA](https://doi.org/10.5281/zenodo.13789584)** — single transient from a CFD model (Introini et al., 2018), SVD-compressed.

```bash
uv run python Code/download_datasets.py --files TRIGA
```

Data path in notebooks: `$NUSHRED_DATA_DIR/TRIGA/`.

## Requirements

```bash
uv sync
```

## Minimum path

To reproduce the main paper result: download **TRIGA**, then run **`01_shred_synthetic.ipynb`** only.

## Notebooks

| Notebook | Description |
| -------- | ----------- |
| `01_shred_synthetic.ipynb` | **Main paper notebook** — constrained sensing and SHRED reconstruction on synthetic TRIGA data |
