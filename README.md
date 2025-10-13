This repository contains the code used for generating the results and plots in https://arxiv.org/abs/2509.13946.

The folder results contains scripts for generating the plots.
he_time_evolution.py is the code for computing the time dynamics of the system.
hold_sweep.py is a script used for sweeping over different hold times in an efficient way (described by fig. 2 in the paper).

The code was ran on a Nvidia GPU using [JAX](https://docs.jax.dev/en/latest/index.html). All packages used are listed in requirements.txt.
