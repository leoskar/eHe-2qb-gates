import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.optimize import minimize
from IPython.display import display, Math, Latex

import sys

sys.path.append("..")
from he_time_evolution import *

"""
Utility functions for analyzing and plotting results from hold time sweeps.
"""


sqrt_iSWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
        [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
)
CZ = np.diag([1, 1, 1, -1])

U_0 = {"II": sqrt_iSWAP, "III": CZ}


def get_U(a1, a2, a3):
    return np.diag(
        np.exp(
            0.5j
            * np.array(
                [2 * a1 - a2 - a3, 2 * a1 - a2 + a3, 2 * a1 + a2 - a3, 2 * a1 + a2 + a3]
            )
        )
    )


def phase_corr_angles(overlaps):
    """
    Given the overlaps matrix, return the angles a1, a2, a3
    for the phase correction matrix which makes all elements except the last real.
    """

    P = np.angle(overlaps)
    a1 = 0.5 * (P[1, 1] + P[2, 2])
    a2 = P[2, 2] - P[0, 0]
    a3 = P[1, 1] - P[0, 0]
    return -a1, -a2, -a3


def phase_correction_mat(overlaps):
    a1, a2, a3 = phase_corr_angles(overlaps)
    U = get_U(a1, a2, a3)
    return U


def fidelity(U, U_0):
    M = U_0.T.conj() @ U
    n = U.shape[0]
    return ((np.trace(M @ M.T.conj()) + np.abs(np.trace(M)) ** 2) / (n * (n + 1))).real


def opt_fidelity(U, config):
    """
    Optimize single-qubit Z-rotations to maximize the fidelity of a two-qubit gate U
    with respect to the ideal gate U_0 (either sqrt(iSWAP) or CZ)
    """

    a1, a2, a3 = phase_corr_angles(U)
    opt_res = minimize(
        lambda x: -fidelity_opt_correction(x, U, partial(fidelity, U_0=U_0[config])),
        [(a2) % (2 * np.pi), (a3) % (2 * np.pi)],
    )

    G = SQ_rotation(opt_res.x, U)
    G = np.exp(-1j * np.angle(G[0, 0])) * G

    return -opt_res.fun, opt_res.x, G


def SQ_rotation(alphas, U):
    a2, a3 = alphas
    P = np.diag(
        [
            np.exp(-0.5j * (a2 + a3)),
            np.exp(-0.5j * (a2 - a3)),
            np.exp(0.5j * (a2 - a3)),
            np.exp(0.5j * (a2 + a3)),
        ]
    )
    return P @ U


def fidelity_opt_correction(alphas, U, fidelity):
    return fidelity(SQ_rotation(alphas, U))


def bmatrix(a, nd=3):
    """Returns a LaTeX bmatrix"""
    text = r"\begin{bmatrix}"
    for row in a:
        text += " & ".join(f"{elem:.{nd}f}" for elem in row) + r" \\ "
    text = text[:-3] + r"\end{bmatrix}"
    return text


class DataMats:
    def __init__(
        self,
        overlaps,
        gate_mats,
        phase_corrections,
        opt_gate_mats,
        opt_phase_corrections,
        fid,
        corrected_fid,
        opt_fid,
        sigmas,
        hold_times,
    ):
        self.overlaps = overlaps
        self.gate_mats = gate_mats
        self.phase_corrections = phase_corrections
        self.opt_gate_mats = opt_gate_mats
        self.opt_phase_corrections = opt_phase_corrections
        self.fid = fid
        self.corrected_fid = corrected_fid
        self.opt_fid = opt_fid
        self.sigmas = sigmas
        self.hold_times = hold_times


class FidelityPlotter:
    def __init__(self, params, config, vmin=0.0):
        self.params = params
        self.config = config

        self.vmin = vmin

        self.fidelity_func = partial(fidelity, U_0=U_0[self.config])

        self.coarse_mats = None

    def get_coarse_mats(self):
        if self.params == "beta" and self.config == "III":
            sigmas = np.arange(0.05, 0.705, 0.01)
        elif self.params == "zz" and self.config == "III":
            sigmas = np.arange(0.05, 0.51, 0.05)
        else:
            sigmas = np.arange(0.05, 0.501, 0.01)

        if self.params in ["zz", "zz_III"] and self.config == "III":
            hold_times = np.arange(0.0, 10.05, 0.1)
        else:
            hold_times = np.arange(0.0, 5.05, 0.1)

        parent_dir = f"{self.params}_params/hold_sweep/config_{self.config}/"

        final_overlaps = np.zeros(
            (len(sigmas), len(hold_times), 4, 4), dtype=np.complex128
        )
        gate_mats = np.zeros((len(sigmas), len(hold_times), 4, 4), dtype=np.complex128)
        opt_gate_mats = np.zeros(
            (len(sigmas), len(hold_times), 4, 4), dtype=np.complex128
        )

        fid = np.zeros((len(sigmas), len(hold_times)))
        corrected_fid = np.zeros((len(sigmas), len(hold_times)))
        opt_fid = np.zeros((len(sigmas), len(hold_times)))

        phase_corrections = np.zeros((len(sigmas), len(hold_times), 3))
        opt_phase_corrections = np.zeros((len(sigmas), len(hold_times), 2))

        for i, sigma in enumerate(sigmas):
            dir = (
                parent_dir
                + f"{hold_times[0]:.2f}-{hold_times[-1]:.2f}-0.10_sigma_{sigma:.2f}/"
            )

            for j, t in enumerate(hold_times):
                _, _, overlaps, _ = np.load(dir + f"t_hold_{t:.2f}.npz").values()
                final_overlaps[i, j] = overlaps[-1, [0, 1, 2, 4]]
                a1, a2, a3 = phase_corr_angles(final_overlaps[i, j])
                phase_corrections[i, j] = np.array([a1, a2, a3])
                U = get_U(a1, a2, a3)
                gate_mats[i, j] = U @ final_overlaps[i, j]
                opt_correction = minimize(
                    lambda x: -fidelity_opt_correction(
                        x, final_overlaps[i, j], self.fidelity_func
                    ),
                    [(a2) % (2 * np.pi), (a3) % (2 * np.pi)],
                    # bounds=[(0, 2 * np.pi), (0, 2 * np.pi)],
                ).x
                opt_phase_corrections[i, j] = opt_correction
                opt_gate_mats[i, j] = SQ_rotation(opt_correction, final_overlaps[i, j])

                fid[i, j] = self.fidelity_func(final_overlaps[i, j])
                corrected_fid[i, j] = self.fidelity_func(gate_mats[i, j])
                opt_fid[i, j] = self.fidelity_func(opt_gate_mats[i, j])

        self.coarse_mats = DataMats(
            final_overlaps,
            gate_mats,
            phase_corrections,
            opt_gate_mats,
            opt_phase_corrections,
            fid,
            corrected_fid,
            opt_fid,
            sigmas,
            hold_times,
        )
        return self.coarse_mats

    def _get_best_gate_text_coarse(self, nd=3):
        texts = []
        ind = np.unravel_index(
            self.coarse_mats.fid.argmax(), self.coarse_mats.fid.shape
        )
        n = np.exp(-1j * np.angle(self.coarse_mats.overlaps[ind][0, 0]))
        texts.append(
            rf"Best fidelity: {self.coarse_mats.fid[ind]:.5f}, ($\sigma$ = {self.coarse_mats.sigmas[ind[0]]:.2f}, "
            + r"$t_{hold}$ = "
            + f"{self.coarse_mats.hold_times[ind[1]]:.2f})\n"
            rf"${bmatrix((n * self.coarse_mats.overlaps[ind]).round(nd), nd)}$"
        )

        ind = np.unravel_index(
            self.coarse_mats.corrected_fid.argmax(),
            self.coarse_mats.corrected_fid.shape,
        )
        n = np.exp(-1j * np.angle(self.coarse_mats.gate_mats[ind][0, 0]))
        texts.append(
            rf"Best fidelity: {self.coarse_mats.corrected_fid[ind]:.5f}, ($\sigma$ = {self.coarse_mats.sigmas[ind[0]]:.2f}, "
            + r"$t_{hold}$ = "
            + f"{self.coarse_mats.hold_times[ind[1]]:.2f})\n"
            rf"${bmatrix((n * self.coarse_mats.gate_mats[ind]).round(nd), nd)}$"
        )

        ind = np.unravel_index(
            self.coarse_mats.opt_fid.argmax(), self.coarse_mats.opt_fid.shape
        )
        n = np.exp(-1j * np.angle(self.coarse_mats.opt_gate_mats[ind][0, 0]))
        texts.append(
            rf"Best fidelity: {self.coarse_mats.opt_fid[ind]:.5f}, ($\sigma$ = {self.coarse_mats.sigmas[ind[0]]:.2f}, "
            + r"$t_{hold}$ = "
            + f"{self.coarse_mats.hold_times[ind[1]]:.2f})\n"
            rf"${bmatrix((n * self.coarse_mats.opt_gate_mats[ind]).round(nd), nd)}$"
        )

        return texts

    def get_best_gate_text_polar(self, nd=2):
        """Returns a LaTeX pmatrix"""

        ind = np.unravel_index(
            self.coarse_mats.opt_fid.argmax(), self.coarse_mats.opt_fid.shape
        )
        n = np.exp(-1j * np.angle(self.coarse_mats.opt_gate_mats[ind][0, 0]))
        array = n * self.coarse_mats.opt_gate_mats[ind]

        text = r"\begin{pmatrix}"
        for row in array:
            text += (
                " & ".join(
                    (
                        rf"{np.abs(elem):.{nd}f}"
                        if (np.abs(elem) > 10 ** (-nd))
                        else rf"0"
                    )
                    + (
                        rf"e^{{ {np.angle(elem)/np.pi:.2f} \pi i}}"
                        if (np.abs(elem) > 10 ** (-nd))
                        and (np.abs((np.angle(elem) / np.pi)) > 1e-2)
                        else rf""
                    )
                    for elem in row
                )
                + r" \\ "
            )
        text = text[:-3] + r"\end{pmatrix}"

        return text
