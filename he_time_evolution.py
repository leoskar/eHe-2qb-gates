import math
import functools
import pint
import jax
import scipy.interpolate

import numpy as np
import jax.numpy as jnp
import sinc_dvr.func_basis as sdf

from opt_einsum import contract
from scipy.special import factorial
from jax import jit
from functools import partial
from integrators import *
from pyscf.lib.linalg_helper import davidson
from time import ctime
from pathlib import Path

PATH = Path(__file__).parent
ureg = pint.UnitRegistry()

# Defining the length and the energy scale from a fixed kappa
# This is the scale we will be using
KAPPA = 2326
length = (
    KAPPA * (4 * ureg.pi * ureg.epsilon_0 * ureg.hbar**2) / (ureg.m_e * ureg.e**2)
).to(ureg.nm)
epsilon = (ureg.hbar**2 / (ureg.m_e * length**2)).to(ureg.microeV)

# Defining the time scale
time_unit = (ureg.hbar / epsilon).to(ureg.ps)
# Conversion from the energy unit to GHz
energy_frequency = (epsilon / ureg.planck_constant).to(ureg.GHz)


@jit
def shielded_coulomb(x_1, x_2, kappa=KAPPA, a=0.01):
    """Shielded Coulomb interaction between two electrons, one at x_1 and one at x_2."""
    return kappa / jnp.sqrt((x_1 - x_2) ** 2 + a**2)



def get_anodes(epsilon=epsilon):
    """Read in the anode basis functions from file and 
    return as a dictionary of interpolating functions."""
    data = np.loadtxt(PATH / "anodic_well_basis.txt")

    # Make the anodes dimensionless
    data[:, 1:] *= epsilon.to(ureg.eV).magnitude
    # Scale by the largest value when flipping the anodes
    # max_val = np.max(data[:, 1:])
    max_val = 0

    return {
        # Make the anodes "smoother" by using interpolation
        f"V{i}": scipy.interpolate.interp1d(data[:, 0], max_val - data[:, i])
        for i in range(1, data.shape[1])
    }


def trans_energies(energies):
    """Convert energies to GHz and make relative to the ground state."""
    return (energies - energies[0]) * energy_frequency


@jit
def ramp_func(t, l_p, l_i, t_r, t_g, sigma, t_start):
    """Ramping function from l_p to l_i, with ramp time t_r, gate time t_g,
    ramp time given by sigma, and starting at t_start.
    """
    # Ramping function from:
    # https://doi.org/10.1103/PhysRevA.87.022309
    return l_p + (l_i - l_p) / 2.0 * (
        jax.scipy.special.erf((t + t_start - 0.5 * t_r) / (jnp.sqrt(2) * sigma))
        - jax.scipy.special.erf((t + t_start - t_g + 0.5 * t_r) / (jnp.sqrt(2) * sigma))
    )


class HeliumTimeEvolver:
    """Class for setting up and performing time-evolution of the two-electron
    Hamiltonian for electrons on superfluid helium.
    """

    def __init__(self, num_states=6, x_lims=(-16, 16), dx=0.1, parameters="beta"):
        # Number of states to evolve
        self.num_states = num_states

        # Define grid
        self.x_min, self.x_max = x_lims
        self.dx = dx
        inds_min = int(self.x_min / dx)
        inds_max = int(self.x_max / dx)  # + 1

        inds_l = jnp.arange(inds_min, 0)
        inds_r = jnp.arange(0, inds_max)
        self.shape = (len(inds_l), len(inds_r))

        x_l = inds_l * dx
        x_r = inds_r * dx

        # Set up kinetic matrix elements
        self.t_l = sdf.setup_t_1d(inds_l[:, None], inds_l[None, :], dx)
        self.t_r = sdf.setup_t_1d(inds_r[:, None], inds_r[None, :], dx)

        # Prepare for potential elements
        self.anodes = get_anodes()
        self.a_l = {key: pot(x_l) for key, pot in self.anodes.items()}
        self.a_r = {key: pot(x_r) for key, pot in self.anodes.items()}

        # Set up interaction elements
        self.u_lr = shielded_coulomb(x_l[:, None], x_r[None, :])

        # Default to stay in config I
        self.ramp_func = lambda t: 0

        # Set voltage function and configuration
        self.parameters = parameters
        self.config_func = self.get_config_func()

    def get_config_func(self):
        """Return a function for the parameterization lambda(t)"""
        if self.parameters == "beta":
            # Parameters from https://doi.org/10.1103/PRXQuantum.5.030324
            config_1 = dict(
                V1=0.29744593869303876 * ureg.eV,
                V2=0.15232776400452422 * ureg.eV,
                V3=0.3443604741610698 * ureg.eV,
                V4=-0.3534934485287361 * ureg.eV,
                V5=0.34592690085141264 * ureg.eV,
                V6=0.14394156388673207 * ureg.eV,
                V7=0.30223900588749886 * ureg.eV,
            )

            config_3 = dict(
                V1=0.2951710788904166 * ureg.eV,
                V2=0.15822548876701445 * ureg.eV,
                V3=0.34056146490338995 * ureg.eV,
                V4=-0.35604868077561891 * ureg.eV,
                V5=0.34445804070646519 * ureg.eV,
                V6=0.14359533828400745 * ureg.eV,
                V7=0.30349197961582158 * ureg.eV,
            )

            def configs(l):
                return {
                    key: (
                        ((1 - l) * config_1[key] + l * config_3[key])
                        / epsilon.to(ureg.eV)
                    ).magnitude
                    for key in sorted(config_1)
                }

            return jit(configs)

        # Zeta parameters from this work for configurations II
        if self.parameters == "zz_II":
            config_1 = dict(
                V1=0.38949688108192176 * ureg.eV,
                V2=0.20070090163982218 * ureg.eV,
                V3=0.40035923698043258 * ureg.eV,
                V4=-0.29061060460856081 * ureg.eV,
                V5=0.39858643090910029 * ureg.eV,
                V6=0.20014667503141512 * ureg.eV,
                V7=0.38139618217562754 * ureg.eV,
            )

            config_2 = dict(
                V1=0.38868156848184054 * ureg.eV,
                V2=0.20669148719921954 * ureg.eV,
                V3=0.40488289793493987 * ureg.eV,
                V4=-0.28836937532658063 * ureg.eV,
                V5=0.40103979818080265 * ureg.eV,
                V6=0.19298026235733345 * ureg.eV,
                V7=0.38256539076273932 * ureg.eV,
            )

            def configs(l):
                return {
                    key: (
                        ((1 - l) * config_1[key] + l * config_2[key])
                        / epsilon.to(ureg.eV)
                    ).magnitude
                    for key in sorted(config_1)
                }

            return jit(configs)
        
        # Zeta parameters from this work for configuration III
        if self.parameters == "zz_III":
            config_1 = dict(
                V1=0.38949688108192176 * ureg.eV,
                V2=0.20070090163982218 * ureg.eV,
                V3=0.40035923698043258 * ureg.eV,
                V4=-0.29061060460856081 * ureg.eV,
                V5=0.39858643090910029 * ureg.eV,
                V6=0.20014667503141512 * ureg.eV,
                V7=0.38139618217562754 * ureg.eV,
            )

            config_3 = dict(
                V1=0.38816593559558416 * ureg.eV,
                V2=0.19400533783311505 * ureg.eV,
                V3=0.40186506472447742 * ureg.eV,
                V4=-0.28909984833118635 * ureg.eV,
                V5=0.39895293947765437 * ureg.eV,
                V6=0.19882044051422076 * ureg.eV,
                V7=0.38243576244420974 * ureg.eV,
            )

            def configs(l):
                return {
                    key: (
                        ((1 - l) * config_1[key] + l * config_3[key])
                        / epsilon.to(ureg.eV)
                    ).magnitude
                    for key in sorted(config_1)
                }

            return jit(configs)

        raise ValueError(
            f"Unrecognized parameter set {self.parameters}. Valid options are beta, zz_II, or zz_III."
        )

    def v(self, t, anodes):
        """Potential at time t given the anode basis functions."""
        voltages = self.config_func(self.ramp_func(t))
        return sum([voltages[key] * anodes[key] for key in sorted(anodes)])

    def set_ramp_func(
        self, t_hold, t_ramp=4 * jnp.sqrt(2), t_start=0.01, sigma=0.3, config="II"
    ):
        """Set the ramping function (lambda(t)) to be used in the time-evolution."""

        if config not in ["II", "III"]:
            raise ValueError(
                f"Unrecognized configuration {config}. Valid options are II or III."
            )

        if self.parameters == "beta":
            if config == "II":
                l_i = 0.455
            elif config == "III":
                l_i = 1.0
        if self.parameters == "zz_II":
            if config == "II":
                l_i = 1.0
            elif config == "III":
                raise ValueError(f"Parameters zz_II only supports configuration II.")
        if self.parameters == "zz_III":
            if config == "II":
                raise ValueError(f"Parameters zz_III only supports configuration III.")
            elif config == "III":
                l_i = 1.0

        sigma = (sigma * ureg.ns / time_unit).to(ureg.dimensionless)
        t_ramp = t_ramp * sigma
        t_start = -(t_start * ureg.ns / time_unit).to(ureg.dimensionless)
        t_hold = (t_hold * ureg.ns / time_unit).to(ureg.dimensionless)
        t_gate = t_hold + 2 * t_ramp

        self.ramp_func = jit(
            partial(
                ramp_func,
                l_p=0,
                l_i=l_i,
                t_r=t_ramp.magnitude,
                t_g=t_gate.magnitude,
                sigma=sigma.magnitude,
                t_start=t_start.magnitude,
            )
        )

    def set_custom_ramp_func(self, f):
        """
        Set a custom ramping function (lambda(t)) to be used in the time-evolution. (Not used in this work.)
        """
        self.ramp_func = f

    def get_H(self):
        """Return the Hamiltonian function H(t, c)"""
        v_l = jit(partial(self.v, anodes=self.a_l))
        v_r = jit(partial(self.v, anodes=self.a_r))
        v_zero = jnp.min(v_r(0)) + jnp.min(v_l(0)) + self.u_lr[0, 0]

        def H(
            t,
            c,
        ):
            c = c.reshape(self.shape)

            return (
                contract("ip, pj -> ij", self.t_l, c, backend="jax")
                + contract("jp, ip -> ij", self.t_r, c, backend="jax")
                + contract("i, ij -> ij", v_l(t), c, backend="jax")
                + contract("j, ij -> ij", v_r(t), c, backend="jax")
                + contract("ij, ij -> ij", self.u_lr, c, backend="jax")
                - v_zero * c
            ).ravel()

        return jit(H)

    @partial(jit, static_argnums=0)
    def ip(self, c_conj, c):
        """Inner product of two states expressed in the sinc-DVR basis."""
        return c_conj @ c * self.dx

    @partial(jit, static_argnums=0)
    def normalize(self, c):
        """Normalize a state expressed in the sinc-DVR basis."""
        return c / jnp.sqrt(self.ip(c.conj(), c))

    def get_orthonormalizer(self, states):
        """Return a function that orthonormalizes a state c against a list of states."""
        k = len(states)

        if k == 0:
            return self.normalize

        states = [*states, jnp.zeros_like(states[0])]

        def orthonormalize(c, states=jnp.array(states), k=k):
            states = states.at[k].set(c)
            b = jnp.zeros(k + 1)
            b = b.at[-1].set(1)
            return self.normalize(jnp.linalg.lstsq(states[: k + 1], b)[0])

        return jit(orthonormalize)

    def run_configI_CI(self, dtau=1e-3):
        """Run CI calculation to get the configuration I states."""
        final_energies, final_states = self.run_CI(dtau=dtau, t0=0)
        self.configI_states = final_states

        return final_energies, final_states

    def run_CI(self, dtau=1e-3, t0=0):
        """Run CI calculation to get the lowest eigenstates at time t0."""
        # Get a good initial guess for the states using imaginary time-propagation
        states = self.imag_time_prop(dtau, t0)
        # Perform Davidson diagonalization to find the lowest eigenstates
        _, davidson_states = self.davidson(states, t0)

        H = self.get_H()

        # Make sure the states are normalized and get their energies
        final_states = [self.normalize(c) for c in davidson_states]
        final_energies = [self.ip(c.conj(), H(t0, c)) for c in final_states]

        return final_energies, final_states

    def save_configI_states_to_file(self, filename=None):
        """
        Save the configuration I states to file.
        """
        if filename == None:
            filename = (
                PATH / f"configI_states_{self.dx}_dx_{self.parameters}_params.npy"
            )
        np.save(filename, self.configI_states)

    def read_configI_states_from_file(self, filename=None):
        """
        Read the configuration I states from file.
        """
        if filename == None:
            if self.parameters in ["zz_II", "zz_III"]:
                params = "zz"
            else:
                params = self.parameters
            
            filename = (
                PATH / f"configI_states_{self.dx}_dx_{params}_params.npy"
            )
        self.configI_states = list(np.load(filename))

    def imag_time_prop(self, dtau, t0=0):
        """Run imaginary time-propagation to get a set of initial states for the Davidson diagonalization."""

        H = self.get_H()

        # Right-hand side for the imaginary time propagation
        rhs_imag = jit(lambda t, c: -H(t0, c))
        # Solver for the linear system
        solver = lambda A, b, x0: jit(
                partial(jax.scipy.sparse.linalg.cg, tol=1e-6),
                static_argnums=(0,),
            )(A, b, x0)[0]

        # Integrator from https://github.com/Schoyen/integrators
        integrate = get_crank_nicolson_solver(rhs_imag, dtau, solver=solver)

        # Run the imaginary time propagation
        states = []
        for _ in range(self.num_states):

            ort_norm = self.get_orthonormalizer(states)
            c = ort_norm(jnp.ones(math.prod(self.shape)).astype(complex))

            E_prev = self.ip(c.conj(), H(t0, c))
            tau = 0
            for _ in range(10000):
                tau, c = integrate(tau, c)
                c = ort_norm(c)
                E_new = self.ip(c.conj(), H(t0, c))

                if abs(E_prev - E_new) < 1e-15:
                    break

                E_prev = E_new

            states.append(c)

        return states
    
    def davidson(self, states, t0=0):
        """Wrapper for the Davidson algorithm from pyscf."""
        H = self.get_H()
        return davidson(
            lambda c: H(t0, c),
            states,
            lambda dx, e, xo: dx,
            nroots=self.num_states,
            max_cycle=int(1e4),
        )

    def evolve(
        self,
        t_start,
        t_stop,
        dt=1e-2,
        initial_states=None,
        integrator="cn",
        solver_tol=1e-5,
        H=None,
    ):
        """Perform time-evolution from t_start to t_stop with time step dt."""

        if initial_states == None:
            initial_states = self.configI_states
            # Only include the "computational" basis states in the time-evolution
            cs = initial_states[:3] + [initial_states[4]]
        else:
            cs = initial_states

        if H is None:
            H = self.get_H()

        rhs = jax.jit(lambda t, c: -1j * H(t, c))

        if integrator == "cn":
            # Define linear solver for the Crank-Nicolson method
            solver = lambda A, b, x0: jit(
                partial(jax.scipy.sparse.linalg.bicgstab, tol=solver_tol),
                static_argnums=(0,),
            )(A, b, x0)[0]

            # Integrator from https://github.com/Schoyen/integrators
            integrate = get_crank_nicolson_solver(rhs, dt, solver=solver)

        elif integrator == "rk4":
            # Not used in this work
            integrate = get_runge_kutta_4_solver(rhs, dt)

        else:
            raise ValueError(
                f"Unrecognized integrator {integrator}. Valid options are cn or rk4."
            )

        # Define inital values
        num_timesteps = int((t_stop - t_start) / dt) + 1

        ts = t_start * np.ones(len(cs))

        time_points = np.zeros((num_timesteps, len(cs)))
        energies = np.zeros((num_timesteps, len(cs)), dtype=initial_states[0].dtype)
        overlaps = np.zeros(
            (num_timesteps, self.num_states, len(cs)), dtype=initial_states[0].dtype
        )

        ts = np.array(ts)
        cs = np.array(cs)

        time_points[0] = ts
        energies[0] = np.array([self.ip(c.conj(), H(t, c)) for t, c in zip(ts, cs)])

        # Perform the time-evolution
        for k, c in enumerate(cs):
            for l, state in enumerate(self.configI_states):
                overlaps[0, l, k] = self.ip(c.conj(), state)

        for i in range(1, num_timesteps):
            ts, cs = zip(*[integrate(t, c) for t, c in zip(ts, cs)])
            ts = np.array(ts)
            cs = np.array(cs)

            time_points[i] = ts
            energies[i] = np.array([self.ip(c.conj(), H(t, c)) for t, c in zip(ts, cs)])
            for k, c in enumerate(cs):
                for l, state in enumerate(self.configI_states):
                    overlaps[i, l, k] = self.ip(c.conj(), state)

        return time_points, energies, overlaps, cs


def main():
    num_states = 6
    hte = HeliumTimeEvolver(num_states=num_states)
    hte.read_configI_states_from_file()

    config = "II"

    t_start = 0.01
    t_hold = 6.25
    sigma = 0.3

    print(
        f"{ctime()}: Starting time evolution to config {config} with sigma = {sigma} and hold time = {t_hold}",
        flush=True,
    )

    t_ramp = 4 * jnp.sqrt(2) * sigma
    t_gate = ((t_hold + 2 * t_ramp + t_start) * ureg.ns / time_unit).to(
        ureg.dimensionless
    )
    hte.set_ramp_func(t_hold=t_hold, t_start=t_start, sigma=sigma, config=config)

    dt = 0.01
    time_points, energies, overlaps, c_final = hte.evolve(
        0,
        t_gate.magnitude,
        dt=dt,
    )

    np.savez(
        f"results/config_{config}/hold_{t_hold}_sigma_{sigma:.2f}_dt_{dt}.npz",
        time_points=time_points,
        energies=energies,
        overlaps=overlaps,
        c_final=c_final,
    )

    print(
        f"{ctime()}: Finished calculation",
        flush=True,
    )


if __name__ == "__main__":
    main()
