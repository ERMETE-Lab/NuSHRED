# Shallow Recurrent Decoder for Nuclear Reactors applications (NuSHRED)

This repository collects the codes regarding the application of the **Shallow REcurrent Decoder** (SHRED) method to **Nuclear Reactors** systems.

In particular, this repository serves as complementary code to the following paper:

- [P1] S. Riva, C. Introini, A. Cammi, and J. N. Kutz, “Robust State Estimation from Partial Out-Core Measurements with Shallow Recurrent Decoder for Nuclear Reactors,” [ArXiV](https://arxiv.org/), 2024.

The simulation data (compressed) are available on Zenodo:

- [D1] Molten Salt Fast Reactor (MSFR) in the accidental scenario Unprotected Loss Of Fuel Flow (ULOFF): [compressed data](https://zenodo.org/)

---

The SHRED method was first proposed and developed in this paper:

- J. Williams, O. Zahn and J. N. Kutz, Sensing with shallow recurrent decoder networks, arxiv (2023) arXiv:2301.12011

The original code base is here: https://github.com/Jan-Williams/pyshred

The [*pyforce* package](https://github.com/ERMETE-Lab/ROSE-pyforce) is used as support for sensor placements and dimensionality reduction, see [Riva et al. (2024)](https://doi.org/10.1016/j.apm.2024.06.040) and [Cammi et al. (2024)](https://doi.org/10.1016/j.nucengdes.2024.113105).

## Structure of the repository

In the folder `shred`, the modules for the implementation of the Shallow Recurrent Decoder (SHRED) network from [pyshred](https://github.com/Jan-Williams/pyshred) are reported.
On the other hand, the folder `Code` is divided into subfolders corresponding to the papers regarding the application of SHRED to nuclear reactor concepts; the dataset are associated as follows

| | Dataset D1 |
|---|-----|
| P1 | x |
