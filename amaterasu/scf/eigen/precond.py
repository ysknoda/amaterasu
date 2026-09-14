"""Preconditioning for the iterative eigensolvers.

An iterative solver converges at a rate set by the spread of ``H``'s spectrum, and in a
plane-wave basis that spread is enormous: the kinetic energy runs from zero to ``ecut``,
while the eigenvalues of interest are all near the bottom. Left alone, the high-``G``
components of a residual dominate every correction and nothing useful happens.

The cure is to multiply the residual by something close to ``1/(H - eps)``. At large
``G`` the Hamiltonian is essentially kinetic, so ``1/T_G`` would do, but that blows up as
``G -> 0``. Teter, Payne and Allan (PRB 40, 12255 (1989)) interpolate between the two
regimes with a rational function of ``x = T_G / T_band``:

    K(x) = (27 + 18x + 12x^2 + 8x^3) / (27 + 18x + 12x^2 + 8x^3 + 16x^4)

which is 1 at ``x = 0`` and falls off as ``1/(2x)`` for large ``x``. Scaling by the
band's own kinetic energy rather than a global constant is what makes it work across
bands of very different character in the same block.
"""

from __future__ import annotations

import numpy as np

__all__ = ["teter_payne_allan"]


def teter_payne_allan(
    residual: np.ndarray, kinetic: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Precondition a block of residuals.

    Parameters
    ----------
    residual
        Shape ``(nbands, npw)``.
    kinetic
        ``|k+G|^2 / 2`` for the k-point, shape ``(npw,)``.
    reference
        Kinetic energy of each band, shape ``(nbands,)``. Values at or below zero are
        replaced by a small positive number, which happens only for a band that is
        numerically zero.
    """
    reference = np.maximum(np.atleast_1d(np.asarray(reference, dtype=float)), 1e-8)
    x = kinetic[None, :] / reference[:, None]
    polynomial = 27.0 + x * (18.0 + x * (12.0 + 8.0 * x))
    return residual * (polynomial / (polynomial + 16.0 * x**4))
