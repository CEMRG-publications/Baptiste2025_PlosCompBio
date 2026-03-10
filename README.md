# Bayesian Calibration of Cardiac Simulations via Gaussian Process Emulation

Code accompanying **Baptiste et al. (2025), PLoS Computational Biology**.

This repository implements a pipeline for emulation and calibration of computationally expensive cardiac mechanics simulations. It combines Gaussian Process (GP) emulation, global sensitivity analysis, history matching, and Markov Chain Monte Carlo (MCMC) sampling to efficiently calibrate material parameters of a finite-element heart model to experimental observations.

## Overview

Running high-fidelity cardiac simulations is expensive. This pipeline replaces the simulator with fast GP emulators and uses them to:

1. **Identify influential parameters** via Sobol-based global sensitivity analysis
2. **Filter implausible regions** of parameter space through iterative history matching
3. **Sample the posterior distribution** of calibrated parameters using MCMC

The calibration targets include regional wall displacement and ventricular volume measurements.

## Pipeline

```
gen_data_files.py          # Preprocess simulation outputs into training datasets
gen_exp_files.py           # Prepare experimental/target data files
prep_sims.py               # Generate simulation input files for each wave
     |
     v
run_gsa.py                 # Train GP emulators (k-fold CV) and run Sobol GSA
     |
     v
run_hm.py                  # Iterative history matching (NROY space identification)
     |
     v
run_mcmc.py                # MCMC calibration constrained to NROY space
process_mcmc.py            # Post-process chains (burn-in, thinning)
test_mcmc.py               # Diagnostics and corner plots
     |
     v
results_plots.py           # Publication figures
```

## Repository Structure

```
├── gen_data_files.py              # Data preprocessing
├── gen_exp_files.py               # Experimental data preparation
├── prep_sims.py                   # Simulation preparation
├── run_gsa.py                     # GP training and global sensitivity analysis
├── run_hm.py                      # History matching
├── run_mcmc.py                    # MCMC sampling (emcee)
├── process_mcmc.py                # MCMC post-processing
├── test_mcmc.py                   # MCMC diagnostics and visualization
├── results_plots.py               # Results visualization
├── mcmc_functions.py              # Log-likelihood and log-prior definitions
└── GPErks_library/                # Core library
    ├── GP_functions.py            # GPyTorch model definitions (RBF kernels)
    ├── GPE_ensemble.py            # GP ensemble with normalization
    ├── kfold_cross_validation_training.py  # K-fold CV training
    ├── global_sobol_sensitivity_analysis.py # Sobol indices computation
    ├── gsa_parameters_ranking.py  # Parameter importance ranking
    ├── history_matching.py        # Multi-wave history matching algorithm
    ├── HM_utils.py                # History matching utilities
    ├── mcmc.py                    # MCMC utilities
    ├── mcmc_functions.py          # MCMC helper functions
    ├── file_utils.py              # I/O and parameter conversion
    ├── hm_plotting.py             # History matching visualization
    ├── gsa_plotting.py            # GSA visualization
    ├── figures.py                 # Posterior and output plotting
    └── plotting.py                # General plotting utilities
```

## Methods

- **Gaussian Process Emulation**: RBF-kernel GPs trained with GPyTorch via the [GPErks](https://github.com/stelong/GPErks) framework, validated using k-fold cross-validation
- **Global Sensitivity Analysis**: Sobol first-order and total-effect indices computed via [SALib](https://github.com/SALib/SALib)
- **History Matching**: Iterative wave-based filtering using implausibility measures, implemented with [Historia](https://github.com/stelong/Historia)
- **MCMC Calibration**: Affine-invariant ensemble sampling via [emcee](https://github.com/dfm/emcee), constrained to the Not Ruled Out Yet (NROY) parameter space

## Dependencies

- Python 3
- [GPErks](https://github.com/stelong/GPErks) — GP emulator training and cross-validation
- [Historia](https://github.com/stelong/Historia) — History matching framework
- [gpytorch](https://gpytorch.ai/) / [PyTorch](https://pytorch.org/) — GP model backend
- [emcee](https://emcee.readthedocs.io/) — MCMC ensemble sampler
- [SALib](https://salib.readthedocs.io/) — Sensitivity analysis
- [corner](https://corner.readthedocs.io/) — Posterior visualization
- NumPy, SciPy, pandas, matplotlib, seaborn, scikit-learn, h5py, diversipy, tqdm

## Citation

If you use this code, please cite:

> Baptiste TM, Rodero C, Sillett CP, Strocchi M, Lanyon CW, et al. (2025) Regional heterogeneity in left atrial stiffness impacts passive deformation in a cohort of patient-specific models. *PLOS Computational Biology* 21(11): e1013656. https://doi.org/10.1371/journal.pcbi.1013656
