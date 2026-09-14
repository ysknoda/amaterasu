"""The kinetic energy operator, which is diagonal in the plane-wave basis.

For ``|k+G>`` the kinetic energy is ``|k+G|^2 / 2`` in Hartree atomic units, so applying
the operator is an elementwise multiply and there is nothing to precompute beyond the
squared lengths already carried by the :class:`~amaterasu.basis.gvectors.GSphere`.

That diagonality is what makes a plane-wave basis worth the large number of functions:
the kinetic term -- the one that is awkward for a localized basis -- is exact and free,
and it is also the term that dominates ``H`` at large ``G``, which is why the
preconditioners in :mod:`amaterasu.scf.eigen.precond` are built from it alone.
"""

from __future__ import annotations

import numpy as np

from ..basis.gvectors import GSphere

__all__ = ["apply_kinetic", "band_kinetic_energies"]


def apply_kinetic(coefficients: np.ndarray, sphere: GSphere) -> np.ndarray:
    """``T |psi>`` for one band or a block of bands."""
    return np.asarray(coefficients) * sphere.kinetic_energy


def band_kinetic_energies(coefficients: np.ndarray, sphere: GSphere) -> np.ndarray:
    """``<psi_n|T|psi_n>`` per band, in Hartree, for normalised coefficients."""
    coefficients = np.atleast_2d(np.asarray(coefficients))
    return np.einsum("ng,g->n", np.abs(coefficients) ** 2, sphere.kinetic_energy)
