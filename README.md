This repository contains the code used for the paper
>**Design and dynamics of two-qubit gates with motional states of electrons on helium**\
> *Oskar Leinonen, Jonas B. Flaten, Stian D. Bilek, Øyvind S. Schøyen, Morten Hjorth-Jensen, Niyaz R. Beysengulov, Zachary J. Stewart, Jared D. Weidman, and Angela K. Wilson*\
> [Phys. Rev. A 113, 032624](https://journals.aps.org/pra/abstract/10.1103/kys2-kfdk)

* The folder *results* contains scripts for creating the figures (note that the data has to be generated or downloaded separately).
* *he_time_evolution.py* is the code for computing the dynamics of the system.
* *hold_sweep.py* is a script used for sweeping over different hold times in an efficient way (described by Fig. 2 in the paper).
* *anodic_well_basis.txt* contains coupling functions between the electrodes and electrons calculated using FEM (see Eq. 2 in the paper).
* *configI_states_0.1_dx_beta_params.npy* and *configI_states_0.1_dx_zz_params.npy* contains the eigenstates in configuration I of the $\beta$ and $\zeta$ voltage function, respectively. These were calculated using the `run_configI_CI` function from `HeliumTimeEvolver`in *he_time_evolution.py* and the CPU version of [JAX](https://docs.jax.dev/en/latest/index.html) together with packages listed in *requirements-cpu.txt*.

All other code was ran on a Nvidia GPU using the GPU version of [JAX](https://docs.jax.dev/en/latest/index.html) together with packages listed in *requirements.txt*.
