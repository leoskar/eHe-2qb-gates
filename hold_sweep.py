from he_time_evolution import *
import jax.numpy as jnp
import sys

from time import ctime
from pathlib import Path


def main():

    # Which parameters to use (i.e. which voltage function)
    params = sys.argv[1]
    # Which configuration to use
    config = sys.argv[2]

    # Hold time sweep range
    t_hold_min = float(sys.argv[3])
    t_hold_max = float(sys.argv[4])
    # Step size in the hold time sweep
    t_step = 0.1

    # Which sigma to use for the ramps
    sigma = float(sys.argv[5])

    run_sweep(t_hold_min, t_hold_max, t_step, sigma, params, config)


def run_sweep(t_hold_min, t_hold_max, t_step, sigma, params, config):

    hte = HeliumTimeEvolver(parameters=params)
    hte.read_configI_states_from_file()

    t_start = 0.0
    t_ramp = 4 * jnp.sqrt(2) * sigma

    # When defining t_ramp directly instead:
    #t_ramp = sigma
    #sigma = t_ramp/(4*jnp.sqrt(2))


    print(
        f"{ctime()}: Starting hold sweep from {t_hold_min} to {t_hold_max}, t_step = {t_step}",
        flush=True,
    )

    # Create directory to save results
    # dir_name = f"results/{params}_params/hold_sweep/config_{config}/{t_hold_min:.2f}-{t_hold_max:.2f}-{t_step:.2f}_sigma_{sigma:.2f}"
    # Path(dir_name).mkdir(exist_ok=True, parents=True)

    # Initial time and states
    t_init = 0
    cs_init = None

    # Time step, converted to dimensionless units
    dt = (0.001*ureg.ns/time_unit).to(ureg.dimensionless).magnitude

    for t_hold in np.arange(t_hold_min, t_hold_max + 0.5*t_step, t_step):

        print(f"{ctime()}: Starting t_hold = {t_hold:.2f}", flush=True)

        # Run intermediate calculation. Subtract 100 steps to decrease deviation to next run.
 
        t_intermediate = ((t_hold + t_ramp + t_start) * ureg.ns / time_unit).to(
            ureg.dimensionless
        ) - 100*dt

        hte.set_ramp_func(t_hold=t_hold, t_start=t_start, sigma=sigma, config=config)

        (
            time_points_intermediate,
            energies_intermediate,
            overlaps_intermediate,
            cs_intermediate,
        ) = hte.evolve(
            t_init, t_intermediate.magnitude, dt=dt, initial_states=cs_init
        )

        print(f"{ctime()}: Finished intermediate calculation", flush=True)

        t_gate = ((t_hold + 2 * t_ramp + t_start) * ureg.ns / time_unit).to(
            ureg.dimensionless
        )

        # Ramp down from the intermediate calculation.
        t_init = time_points_intermediate[-1, 0]
        cs_init = list(cs_intermediate)
        time_points, energies, overlaps, cs = hte.evolve(
            t_init, t_gate.magnitude, dt=dt, initial_states=cs_init
        )

        # np.savez(
        #     dir_name + f"/t_hold_{t_hold:.2f}.npz",
        #     time_points=np.concatenate((time_points_intermediate, time_points[1:])),
        #     energies=np.concatenate((energies_intermediate, energies[1:])),
        #     overlaps=np.concatenate((overlaps_intermediate, overlaps[1:])),
        #     cs=cs,
        # )

        print(f"{ctime()}: Finished t_hold = {t_hold:.2f}", flush=True)

    print(f"{ctime()}: Finished hold sweep", flush=True)


if __name__ == "__main__":
    main()
