"""The Hartree potential and energy from a density on the FFT grid.

In reciprocal space Poisson's equation is diagonal,

    V_H(G) = 4 pi n(G) / G^2 ,      E_H = (Omega/2) sum_{G != 0} 4 pi |n(G)|^2 / G^2 ,

with the ``G = 0`` term set to zero. That is not an approximation but a bookkeeping
choice: the ``G = 0`` divergence of the electron-electron repulsion cancels against those
of the electron-ion attraction and the ion-ion repulsion for a neutral cell, and the three
finite remainders are collected in the alpha term of the local potential and the
background term of the Ewald sum (see :mod:`amaterasu.hamiltonian.ewald`).

The energy is taken from the sum over ``|n(G)|^2`` rather than from ``(1/2) integral V_H n``
because that form is manifestly non-negative term by term, so rounding cannot make it
negative for a near-zero density.
"""

from __future__ import annotations

import numpy as np

from ..basis.fft import FFTGrid
from ..basis.gvectors import GSphere

__all__ = ["hartree"]


def hartree(grid: FFTGrid, sphere: GSphere, density: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``(E_H in Hartree, V_H(r) in Hartree)`` for a total density on the grid.

    ``density`` is the *total* electron density (both spins summed); the Hartree term
    does not know about spin.
    """
    density = np.asarray(density, dtype=float)
    if density.shape != tuple(grid.shape):
        raise ValueError(f"density has shape {density.shape}, grid is {grid.shape}")

    coefficients = grid.gather(grid.to_reciprocal(density.astype(complex)), sphere)
    g2 = sphere.g2.copy()
    nonzero = g2 > 1e-12
    potential_g = np.zeros_like(coefficients)
    potential_g[nonzero] = 4.0 * np.pi * coefficients[nonzero] / g2[nonzero]

    energy = (
        0.5
        * grid.volume
        * float(np.sum(4.0 * np.pi * np.abs(coefficients[nonzero]) ** 2 / g2[nonzero]))
    )
    potential = grid.to_real(grid.scatter(potential_g, sphere))
    return energy, np.ascontiguousarray(potential.real)
