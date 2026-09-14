"""The Kohn-Sham Hamiltonian applied to a block of bands.

``H |psi> = T |psi> + V_eff |psi> + V_NL |psi>`` with the three terms living in the three
places that make each of them cheap:

* ``T`` is diagonal in reciprocal space -- an elementwise multiply on the sphere;
* ``V_eff`` (local ionic + Hartree + exchange-correlation) is diagonal in real space, so
  the band is transformed to the grid, multiplied, and transformed back;
* ``V_NL`` is a low-rank update built from the Kleinman-Bylander projectors.

Bands are carried as a block of shape ``(nbands, npw)`` and every step is batched over
that axis, so the FFTs go out in one call and the projector contraction is a single
matrix product. This is the routine the whole calculation spends its time in.

Aliasing
--------
Multiplying by ``V_eff`` on the grid is exact only if the product fits the grid. With the
standard norm-conserving choice ``ecut_rho = 4 ecut`` the product ``V_eff psi`` reaches
``3 G_max``, and the components above the grid's ``2 G_max`` fold back to at least
``G_max`` -- that is, outside the wavefunction sphere, where they are discarded. A smaller
density cutoff would fold them back *inside* the sphere and contaminate ``H|psi>``, which
is the real reason for the factor of four.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinetic import apply_kinetic
from .system import PlaneWaveSystem

__all__ = ["Hamiltonian"]


@dataclass
class Hamiltonian:
    """A Kohn-Sham Hamiltonian at one fixed effective potential."""

    system: PlaneWaveSystem
    #: Local effective potential in Hartree, shape ``(nspin, *grid.shape)``.
    effective_potential: np.ndarray
    #: PAW only: ``H^a_ij = dE/dD_ij``, shape ``(nspin, nproj, nproj)``. Unlike a
    #: norm-conserving ``D_ij``, which comes from the dataset and never moves, this is a
    #: functional of the current potential and is rebuilt every SCF iteration.
    coupling: np.ndarray | None = None

    def __post_init__(self):
        potential = np.asarray(self.effective_potential, dtype=float)
        if potential.ndim == 3:
            potential = potential[None, ...]
        expected = (self.system.nspin, *self.system.grid.shape)
        if potential.shape != expected:
            raise ValueError(
                f"effective potential has shape {potential.shape}, expected {expected}"
            )
        self.effective_potential = potential

    def apply(self, coefficients: np.ndarray, ik: int, spin: int = 0) -> np.ndarray:
        """``H |psi>`` for one band or a block of bands at k-point ``ik``."""
        sphere = self.system.wave_spheres[ik]
        grid = self.system.grid
        result = apply_kinetic(coefficients, sphere)
        real = grid.wave_to_real(coefficients, sphere)
        result = result + grid.real_to_wave(self.effective_potential[spin] * real, sphere)
        coupling = None if self.coupling is None else self.coupling[spin]
        result = result + self.system.projectors[ik].apply(coefficients, coupling)
        return result

    def apply_overlap(self, coefficients: np.ndarray, ik: int) -> np.ndarray:
        """``S |psi>``, the identity unless the calculation is PAW.

        The PAW overlap is a low-rank update of the identity, so applying it costs one
        projector contraction -- much less than applying ``H`` -- which is what makes a
        generalised Davidson iteration affordable.
        """
        paw = self.system.paw
        if paw is None:
            return np.asarray(coefficients)
        return paw.apply_overlap(coefficients, self.system.projectors[ik])

    @property
    def overlap_operator(self):
        """A per-k-point ``S`` for the eigensolver; ``None`` when the basis is orthonormal."""
        if self.system.paw is None:
            return None
        return self.apply_overlap

    def local_energy(self, density: np.ndarray) -> float:
        """``integral V_eff n`` in Hartree, summed over spin channels."""
        density = np.asarray(density, dtype=float)
        return float(
            np.sum(self.effective_potential * density) * self.system.grid.cell_volume_per_point
        )
