# P5: Multi-Fidelity Learning with Shallow Recurrent Decoders

Notebooks supporting:

> Riva, S., Introini, C., Kutz, J. N. & Cammi, A. (2026). Multi-Fidelity Learning with Shallow Recurrent Decoders for Multi-Physics Applications. [arXiv:2606.05202](https://arxiv.org/abs/2606.05202)

Three multi-fidelity test cases map low-fidelity (LF) scalar or reduced models to high-fidelity (HF) spatial fields via MF-SHRED.

## Datasets

| Case                         | Zenodo folder    | Download                                                         |
| ---------------------------- | ---------------- | ---------------------------------------------------------------- |
| Neutronics (LRA benchmark)   | `LRA-neutronics` | `uv run python Code/download_datasets.py --files LRA-neutronics` |
| Reaction–diffusion–advection | `RDA`            | `uv run python Code/download_datasets.py --files RDA`            |
| MSFR (0D DDE → OpenFOAM MP)  | `MSFR`           | `uv run python Code/download_datasets.py --files MSFR`           |

[Datasets on Zenodo](https://doi.org/10.5281/zenodo.13789584)

The Neutronics and RDA cases can also be generated in-notebook with `dolfinx` (see **Requirements** below). Pre-generated archives on Zenodo are sufficient to run the MF-SHRED notebooks.

## Requirements

**Running MF-SHRED notebooks** (pre-generated data from Zenodo):

```bash
uv sync
```

**Regenerating raw data** for Neutronics and RDA requires FEniCSx (`dolfinx` v0.10.0) in a separate conda environment — not available via `uv`:

```bash
conda create -n dolf python=3.10
conda activate dolf
python -m pip install gmsh
conda install -c conda-forge fenics-dolfinx=0.10 mpich pyvista ipykernel scipy
```

Neutronics solvers are adapted from the [OFELIA](https://github.com/ERMETE-Lab) repository (ERMETE Lab). The MSFR case reuses the OpenFOAM dataset from P1/P2 and builds LF trajectories in-notebook (no dolfinx required).

## Minimum path

| Case | Download | Notebook(s) to run |
| ---- | -------- | ------------------- |
| Neutronics | `LRA-neutronics` | `Neutronics/02_mf_shred.ipynb` |
| RDA | `RDA` | `ReactionDiffAdvection/01_mf_shred.ipynb` |
| MSFR | `MSFR` | `MSFR/01_LF_vs_HF_comparison.ipynb`, then `MSFR/02_mfshred.ipynb` |

## Layout and notebooks

### `Neutronics/` — point kinetics → multigroup diffusion (LRA 2D)

| Notebook                  | Description                                                               |
| ------------------------- | ------------------------------------------------------------------------- |
| `01_generate_snaps.ipynb` | Generate diffusion snapshots with dolfinx (optional if using Zenodo data) |
| `02_mf_shred.ipynb`       | **Main neutronics notebook** — MF-SHRED training and evaluation           |

### `ReactionDiffAdvection/` — lumped ODE → coupled PDE species transport

| Notebook                    | Description                                                          |
| --------------------------- | -------------------------------------------------------------------- |
| `00a_create_dataset.ipynb`  | Generate HF/LF datasets with dolfinx (optional if using Zenodo data) |
| `00b_explore_dataset.ipynb` | Explore the pre-generated RDA dataset                                |
| `01_mf_shred.ipynb`         | **Main RDA notebook** — MF-SHRED training and evaluation             |

Supporting Python modules live in `ReactionDiffAdvection/models/`.

### `MSFR/` — 0D circulating-fuel DDE → multi-physics OpenFOAM

| Notebook                       | Description                                                         |
| ------------------------------ | ------------------------------------------------------------------- |
| `01_LF_vs_HF_comparison.ipynb` | Build LF dataset and compare against HF OpenFOAM transients         |
| `02_mfshred.ipynb`             | **Main MSFR notebook** — MF-SHRED from LF measurements to MP fields |
