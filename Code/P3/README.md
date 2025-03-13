# P3: Towards Efficient Parametric State Estimation in Circulating Fuel Reactors with Shallow Recurrent Decoder Networks

This folder collects the supporting notebooks of the following paper:

- C. Introini, S. Riva, J. N. Kutz and A. Cammi, “From Models To Experiments: Shallow Recurrent Decoder Networks on the DYNASTY Experimental Facility,” 2025. preprint available at
[https://arxiv.org/abs/2503.08907](arxiv.org/abs/2503.08907).

On [Zenodo](https://zenodo.org/records/15015236), the simulation data (compressed) are available.

The data are been rescaled in the range [0,1] to ensure an open sharing of them.

The notebook `01_shred_verification_parametric.ipynb` verifies the SHRED using synthetic data only from dataset D3. The notebook `02a_shred_validation_parametric.ipynb` investigates the training with simulation data (parametric scenarios) and later deployment with experimental data. The last notebook `02b_shred_validation_forecasting.ipynb` analyses the prediction capabilities of the SHRED network beyond in time.
