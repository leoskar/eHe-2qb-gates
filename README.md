This repository contains the code used for generating the results and plots in https://arxiv.org/abs/2509.13946.

* The folder *results* contains scripts for generating the plots.
* *he_time_evolution.py* is the code for computing the time dynamics of the system.
* *hold_sweep.py* is a script used for sweeping over different hold times in an efficient way (described by Fig. 2 in the paper).
* *anodic_well_basis.txt* contains coupling functions between the electrodes and electrons calculated using FEM (see Eq. 2 in the paper).
* *configI_states_0.1_dx_beta_params.npy* and *configI_states_0.1_dx_zz_params.npy* contains the eigenstates in configuration I of the $\beta$ and $\zeta$ voltage function, respectively. These were calculated using the `run_configI_CI` function from `HeliumTimeEvolver`in *he_time_evolution.py*.

The code was ran on a Nvidia GPU using [JAX](https://docs.jax.dev/en/latest/index.html). All packages used are listed in *requirements.txt*.
