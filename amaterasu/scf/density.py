"""Build the electron density from occupied Kohn-Sham orbitals.

With ``psi_nk(r) = Omega^(-1/2) sum_G c_nk(G) exp(i(k+G).r)`` and ``sum_G |c|^2 = 1``, the
density is

    n_s(r) = (1/Omega) sum_k w_k sum_n g f_snk |u_snk(r)|^2 ,

``u`` being the FFT of the coefficients onto the grid and ``g = 2/nspin`` the spin
degeneracy of a state. The ``1/Omega`` is the normalisation of the Bloch function; the
plane-wave transform in :mod:`amaterasu.basis.fft` carries no volume factor of its own,
so it appears explicitly here.

Symmetrisation is not optional when the k-sum runs over the irreducible wedge. The wedge
sum produces a density with the symmetry of the wedge rather than of the crystal, and
feeding that back through the SCF loop lowers the symmetry of the converged solution.
:meth:`~amaterasu.hamiltonian.system.PlaneWaveSystem.symmetrize_density` restores it, and
because the ``G = 0`` coefficient is invariant under every operation, the electron count
survives the average exactly.
"""

from __future__ import annotations

import numpy as np

from ..hamiltonian.system import PlaneWaveSystem

__all__ = ["compute_density", "electron_count"]


def electron_count(system: PlaneWaveSystem, density: np.ndarray) -> float:
    """``integral n(r) dr`` over the cell, summed over spins."""
    return float(np.sum(density)) * system.grid.cell_volume_per_point


def compute_density(
    system: PlaneWaveSystem,
    wavefunctions: list[list[np.ndarray]],
    occupations: np.ndarray,
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    """Accumulate the density from the occupied orbitals.

    Parameters
    ----------
    wavefunctions
        ``wavefunctions[spin][ik]`` of shape ``(nbands, npw_k)``, orthonormal rows.
    occupations
        Shape ``(nspin, nkpoints, nbands)``, each in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Shape ``(nspin, *grid.shape)`` in electrons/Bohr^3.
    """
    grid = system.grid
    degeneracy = 2.0 / system.nspin
    density = np.zeros((system.nspin, *grid.shape))

    for spin in range(system.nspin):
        for ik, (sphere, weight) in enumerate(
            zip(system.wave_spheres, system.kpoints.weights, strict=True)
        ):
            factors = degeneracy * weight * occupations[spin, ik]
            active = factors > 1e-12
            if not np.any(active):
                continue
            real = grid.wave_to_real(wavefunctions[spin][ik][active], sphere)
            density[spin] += np.einsum("n,n...->...", factors[active], np.abs(real) ** 2)

    density /= grid.volume
    if symmetrize:
        # Symmetrisation is a linear average over operations that all fix G = 0, so it
        # preserves each channel's electron count exactly. The floor that follows does
        # not -- it removes the small negative excursions the finite basis can leave in
        # the interstitial, where a GGA would otherwise see an unphysical density -- so
        # each channel is rescaled back to the charge it had before clipping.
        symmetrized = system.symmetrize_density(density)
        before = symmetrized.sum(axis=(-3, -2, -1))
        clipped = np.maximum(symmetrized, 0.0)
        after = clipped.sum(axis=(-3, -2, -1))
        scale = np.where(after > 0, before / np.where(after > 0, after, 1.0), 1.0)
        density = clipped * scale[:, None, None, None]
    return density
