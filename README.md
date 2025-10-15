# Shallow Recurrent Decoder for Nuclear Reactors Applications (NuSHRED)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-magenta.svg)](https://www.python.org/)
[![Data](https://img.shields.io/badge/Datasets-10.5281/zenodo.15015236-blue.svg)](https://doi.org/10.5281/zenodo.15015236)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?logo=youtube)](https://www.youtube.com/watch?v=AUuGhojLiFk)

This repository collects the codes regarding the application of the **Shallow REcurrent Decoder** (SHRED) method to **Nuclear Reactors** systems 🏭⚛️

---

## 📄 Related Publications

This repository serves as complementary code to the following papers:

- **[P1]** Riva, S., Introini, C., Cammi, A., & Kutz, J. N. (2025). Robust State Estimation from Partial Out-Core Measurements with Shallow Recurrent Decoder for Nuclear Reactors. *Progress in Nuclear Energy*, vol. 189, pp. 105928  [![arXiv](https://img.shields.io/badge/Ensemble%20SHRED-2409.12550-b31b1b.svg)](https://doi.org/10.1016/j.pnucene.2025.105928)

- **[P2]** Riva, S., Introini, C., Kutz, J. N. & Cammi, A. (2025). Towards Efficient Parametric State Estimation in Circulating Fuel Reactors with Shallow Recurrent Decoder Networks [![arXiv](https://img.shields.io/badge/Parametric%20MSFR-2503.08904-b31b1b.svg)](http://arxiv.org/abs/2503.08904)

- **[P3]** Introini, C., Riva, S., Kutz, J. N. & Cammi, A.(2025). From Models To Experiments: Shallow Recurrent Decoder Networks on the DYNASTY Experimental Facility [![arXiv](https://img.shields.io/badge/Verification&Validation%20DYNASTY-2503.08907-b31b1b.svg)](http://arxiv.org/abs/2503.08907)

- **[P4]** Riva, S., Introini, C., Cammi, A., & Kutz, J. N. (2025). Constrained Sensing and Reliable State Estimation with Shallow Recurrent Decoders on a TRIGA Mark II Reactor. [![arXiv](https://img.shields.io/badge/%20TRIGA-2503.08908-b31b1b.svg)](https://arxiv.org/abs/2510.12368)

---

## 📊 Simulation Data
The compressed simulation datasets are available on **Zenodo**:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15015236.svg)](https://doi.org/10.5281/zenodo.15015236)

- **[D1]** Molten Salt Fast Reactor (MSFR) in the accidental scenario *Unprotected Loss Of Fuel Flow (ULOFF)* - Single Transient (Reconstruction mode)
- **[D2]** Molten Salt Fast Reactor (MSFR) in the accidental scenario *Unprotected Loss Of Fuel Flow (ULOFF)* - Parametric Transients
- **[D3]** DYNASTY Experimental Facility - Single Transient (Reconstruction & Prediction mode) and Parametric Transients
- **[D4]** CFD model of TRIGA Mark II Reactor - Single Transient (Reconstruction mode)

🎥 If you want to know more about the SHRED method for nuclear reactors, check out this [**YouTube video**](https://www.youtube.com/watch?v=AUuGhojLiFk)!

---

## 🏗️ Foundations of SHRED

The SHRED method was first proposed and developed in this paper:

- **J. Williams, O. Zahn and J. N. Kutz**, *Sensing with shallow recurrent decoder networks*, arXiv (2023) [arXiv:2301.12011]

📌 The original code base is available here: [**github.com/Jan-Williams/pyshred**](https://github.com/Jan-Williams/pyshred)

This repository also builds upon a related implementation:

- **Matteo Tomasetto, Jan P. Williams, Francesco Braghin, Andrea Manzoni, J. Nathan Kutz**, *Reduced Order Modeling with Shallow Recurrent Decoder Networks*, arXiv (2025) [arXiv:2502.10930]

📌 Improvements for Parametric datasets are available here: [**github.com/MatteoTomasetto/SHRED-ROM**](https://github.com/MatteoTomasetto/SHRED-ROM)

Additionally, the [*pyforce* package](https://github.com/ERMETE-Lab/ROSE-pyforce) is used for **sensor placements** and **EIM/GEIM comparison** in P1. See:
- [Riva et al. (2024)](https://doi.org/10.1016/j.apm.2024.06.040)
- [Cammi et al. (2024)](https://doi.org/10.1016/j.nucengdes.2024.113105)

---

## 📂 Repository Structure

📁 **shred/** → Modules for the implementation of the SHRED network from [**github.com/Jan-Williams/pyshred**](https://github.com/Jan-Williams/pyshred) and [**github.com/MatteoTomasetto/SHRED-ROM**](https://github.com/MatteoTomasetto/SHRED-ROM)

📁 **Code/** → Subfolders corresponding to the applications of SHRED in nuclear reactor concepts, with datasets associated as follows:

| | MSFR-ULOFF D1 |  MSFR-ULOFF D2  | DYNASTY D3 | TRIGA D4 |
|---|:---:|:---:|:---:| :---:|
| **P1** | ✅ |    |    |    |
| **P2** |    | ✅ |    |    |
| **P3** |    |    | ✅ |    |
| **P4** |    |    |    | ✅ |

## ▶️ How to Execute

1️⃣ **Clone or download** the repository.

2️⃣ **Download the datasets** and move them into the appropriate directory.

3️⃣ **Install the required dependencies:**
   - Main dependencies: *pytorch*, *numpy*, *scikit-learn*, *matplotlib*, *scipy*, *pyvista*.
   - For P1, **pyforce** is required, see [**installation instructions**](https://ermete-lab.github.io/ROSE-pyforce/installation.html#set-up-a-conda-environment-for-pyforce)

Other packages will be installed as part of the requirements.

---

## 📬 Contact Information

For inquiries, please contact:
📧 stefano.riva@polimi.it, carolina.introini@polimi.it, antonio.cammi@polimi.it, kutz@uw.edu.

For **issues** or **bugs**, refer to the **GitHub Issues** section of this repository.

---

## 📊 Results

### 📌 Paper 1

| Fast Flux $\phi_1$ | Temperature $T$ | Velocity $\mathbf{u}$ |
|-------|---|---|
| <img src="media/P1/flux1.gif" width="300"> | <img src="media/P1/T.gif" width="300"> | <img src="media/P1/U.gif" width="300"> |

### 📌 Paper 2

**Out-Core Sensing (Fast Flux)**

| Fast Flux $\phi_1$ | Temperature $T$ | Velocity $\mathbf{u}$ | Precursors Group 1 $c_1$ |
|-------|---|---|---|
| <img src="media/P2/flux1.gif" width="300"> | <img src="media/P2/T.gif" width="300"> | <img src="media/P2/U.gif" width="300"> | <img src="media/P2/prec1.gif" width="300"> |

**Mobile Sensors (Fist Group of Precursors)**

| Fast Flux $\phi_1$ | Temperature $T$ | Velocity $\mathbf{u}$ | Precursors Group 1 $c_1$ |
|-------|---|---|---|
| <img src="media/P2/flux1_mobile_sens.gif" width="300"> | <img src="media/P2/T_mobile_sens.gif" width="300"> | <img src="media/P2/U_mobile_sens.gif" width="300"> | <img src="media/P2/prec1_mobile_sens.gif" width="300"> |

**Mobile Probes (only position measaured)**

| Fast Flux $\phi_1$ | Temperature $T$ | Velocity $\mathbf{u}$ | Precursors Group 1 $c_1$ |
|-------|---|---|---|
| <img src="media/P2/flux1_mobile_probes.gif" width="300"> | <img src="media/P2/T_mobile_probes.gif" width="300"> | <img src="media/P2/U_mobile_probes.gif" width="300"> | <img src="media/P2/prec1_mobile_probes.gif" width="300"> |


### 📌 Paper 3

| **Case**                  | **Visualization**                                             |
|-----------------------------|-------------------------------------------------------------|
| **Parametric Verification** | <img src="media/P3/ParametricVerification.gif" width="400"> |
|-----------------------------|-------------------------------------------------------------|
| **Parametric Validation**   | <img src="media/P3/ParametricValidation.gif" width="400">   |
| **Prediction Validation**   | <img src="media/P3/PredictionValidation.gif" width="400">   |

### 📌 Paper 4

| Temperature $T$ | Velocity $\mathbf{u}$ |
|---|---|
| <img src="media/P4/T.gif" width="300"> | <img src="media/P4/U.gif" width="300"> |
