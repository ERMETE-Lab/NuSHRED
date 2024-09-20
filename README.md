# Shallow Recurrent Decoder for Nuclear Reactors applications (NuSHRED)

This repository collects the codes regarding the application of the **Shallow REcurrent Decoder** (SHRED) method to **Nuclear Reactors** systems.

In particular, this repository serves as complementary code to the following paper:

- [P1] Riva, S., Introini, C., Cammi, A., & Kutz, J. N. (2024). Robust State Estimation from Partial Out-Core Measurements with Shallow Recurrent Decoder for Nuclear Reactors. arXiv [Physics.Ins-Det]. Retrieved from http://arxiv.org/abs/2409.12550

The simulation data (compressed) are available on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13789585.svg)](https://doi.org/10.5281/zenodo.13789585)

- [D1] Molten Salt Fast Reactor (MSFR) in the accidental scenario Unprotected Loss Of Fuel Flow (ULOFF)

---

The SHRED method was first proposed and developed in this paper:

- J. Williams, O. Zahn and J. N. Kutz, Sensing with shallow recurrent decoder networks, arxiv (2023) arXiv:2301.12011

The original code base is here: https://github.com/Jan-Williams/pyshred

The [*pyforce* package](https://github.com/ERMETE-Lab/ROSE-pyforce) is used as support for sensor placements and dimensionality reduction, see [Riva et al. (2024)](https://doi.org/10.1016/j.apm.2024.06.040) and [Cammi et al. (2024)](https://doi.org/10.1016/j.nucengdes.2024.113105).

## Structure of the repository

In the folder `shred`, the modules for the implementation of the Shallow Recurrent Decoder (SHRED) network from [pyshred](https://github.com/Jan-Williams/pyshred) are reported.
On the other hand, the folder `Code` is divided into subfolders corresponding to the papers regarding the application of SHRED to nuclear reactor concepts; the dataset are associated as follows

| | MSFR-ULOFF D1 |
|---|-----|
| P1 | x |

## How to execute
Clone or download the repository, download the correspondent datasets and move in the same directory of the notebooks to execute.

The main requirements to execute the notebooks are *pytorch* and *pyforce*, see instructions [here](https://ermete-lab.github.io/ROSE-pyforce/installation.html#set-up-a-conda-environment-for-pyforce); other packages will be installed as part of the requirements.


## Contact Information

If interested, please contact stefano.riva@polimi.it, carolina.introini@polimi.it, antonio.cammi@polimi.it, kutz@uw.edu.

In case of any problems, refer to Github Issues of this repository.
