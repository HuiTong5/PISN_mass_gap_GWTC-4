# Evidence of the pair instability gap in the distribution of black hole masses (arxiv/2509.04151)
Corresponding author: Hui Tong (hui.tong@monash.edu)

## Description
This repository contains the analysis scripts, models and results.

## Analysis

Please refer to [`analysis-cripts/README.md`](./analysis-scripts/README.md) for instructions on running analyses with some post-processing scripts.

## Models

Models are stored in three directories, 
- `models/default_spin` for analyses using the spin models inherited from the default in `default/default.py`, 
- `models/spin_transition` for analyses using mass-dependent spin models in [Antonini et al.](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.134.011401)),
- `models/pairing_function` where the mass model formalism follows [Farah et al.](https://iopscience.iop.org/article/10.3847/1538-4357/ad0558).

## Data

Posterior samples of GWTC-4 events and sensitivity estimate file used in hierarchical inference can be downloaded following `analysis-scripts/collect_samples.sh`. The data will be saved under `data/`. Our nuclear physics constraint also requires data of [Farmer et al](https://iopscience.iop.org/article/10.3847/2041-8213/abbadd) from [Zenodo](https://zenodo.org/records/3559859).

## Results

Results of this study can be found under `data_release/`, including a notebook used to make the plots in our paper.

