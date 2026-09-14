"""Plane-wave basis: reciprocal lattice vectors within an energy cutoff.

Conventions
-----------
Everything here is in Hartree atomic units: lengths in Bohr, reciprocal vectors in
1/Bohr with the 2*pi convention (``b_i . a_j = 2*pi delta_ij``), energies in Hartree.
The kinetic energy of the plane wave ``exp(i (k+G).r)`` is ``|k+G|^2 / 2``, so a cutoff
``ecut`` admits every vector with

    |k + G| <= sqrt(2 * ecut).

Note how this reads in the Rydberg convention used by Quantum ESPRESSO and pslibrary:
since 1 Ha = 2 Ry, the same condition is ``|k+G|^2 <= ecut[Ry]``, which is why cutoffs
quoted in Ry are numerically twice those quoted in Ha.

Two spheres appear in a calculation:

* a **wavefunction sphere** per k-point, holding ``|k+G|^2/2 <= ecut``;
* a **density sphere** at Gamma, holding ``|G|^2/2 <= ecut_rho``.

They are not independent. The density is ``|psi|^2``, whose Fourier components reach twice
the largest wavefunction G, so representing it without aliasing needs
``ecut_rho = 4 * ecut`` for norm-conserving pseudopotentials. PAW augmentation charges are
harder still and need a larger ratio, taken from the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GSphere", "estimate_count", "miller_bounds", "reciprocal_cell"]


def reciprocal_cell(cell: np.ndarray) -> np.ndarray:
    """Reciprocal lattice vectors (rows, 1/Bohr) for direct ``cell`` (rows, Bohr).

    Uses the 2*pi convention, so ``b_i . a_j = 2*pi delta_ij``.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be 3x3, got {cell.shape}")
    volume = float(np.linalg.det(cell))
    if abs(volume) < 1e-12:
        raise ValueError(f"cell is singular (volume {volume:.3e} Bohr^3)")
    return 2.0 * np.pi * np.linalg.inv(cell).T


def miller_bounds(cell: np.ndarray, gmax: float) -> np.ndarray:
    """Smallest per-direction Miller-index bounds enclosing the sphere ``|G| <= gmax``.

    Since ``G . a_i = 2*pi h_i``, any ``G`` inside the sphere obeys
    ``|h_i| <= |G| |a_i| / (2*pi) <= gmax |a_i| / (2*pi)``. That bound is rigorous, so the
    enclosing box never clips the sphere -- it is merely not tight for skewed cells.
    """
    lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
    return np.ceil(gmax * lengths / (2.0 * np.pi)).astype(int)


def estimate_count(volume: float, ecut: float) -> float:
    """Asymptotic number of plane waves below ``ecut`` (Ha) in a cell of ``volume`` (Bohr^3).

    The reciprocal-space density of G-points is ``volume / (2*pi)^3``, so the count inside
    a sphere of radius ``sqrt(2*ecut)`` approaches

        N = volume * (2*ecut)^(3/2) / (6*pi^2).

    The true count fluctuates about this as whole shells of vectors cross the cutoff, so
    it is an asymptote rather than an identity.
    """
    return volume * (2.0 * ecut) ** 1.5 / (6.0 * np.pi**2)


@dataclass
class GSphere:
    """The set of plane waves inside an energy cutoff, sorted by kinetic energy.

    Sorting matters: it puts the low-energy, physically dominant components first, makes
    the layout independent of how the enclosing box was scanned, and lets a smaller sphere
    be taken as a prefix of a larger one.
    """

    #: Integer Miller indices ``(h, k, l)``, shape ``(npw, 3)``.
    miller: np.ndarray
    #: Cartesian ``k+G`` in 1/Bohr, shape ``(npw, 3)``.
    cartesian: np.ndarray
    #: ``|k+G|^2`` in 1/Bohr^2, shape ``(npw,)``.
    g2: np.ndarray
    #: Fractional k-point this sphere was built around (zeros for a density sphere).
    kpoint: np.ndarray
    #: Cutoff in Hartree.
    ecut: float
    #: Direct cell in Bohr (rows).
    cell: np.ndarray

    def __len__(self) -> int:
        return len(self.miller)

    @property
    def kinetic_energy(self) -> np.ndarray:
        """Kinetic energy ``|k+G|^2 / 2`` of each plane wave, in Hartree."""
        return 0.5 * self.g2

    @property
    def gmax(self) -> float:
        """Cutoff radius ``sqrt(2*ecut)`` in 1/Bohr."""
        return float(np.sqrt(2.0 * self.ecut))

    def __str__(self) -> str:
        volume = abs(float(np.linalg.det(self.cell)))
        expected = estimate_count(volume, self.ecut)
        return (
            f"{len(self)} plane waves at ecut={self.ecut:.4f} Ha "
            f"({2 * self.ecut:.4f} Ry), asymptotic estimate {expected:.1f}"
        )

    @classmethod
    def build(
        cls, cell: np.ndarray, ecut: float, kpoint=(0.0, 0.0, 0.0), *, sort: bool = True
    ) -> GSphere:
        """Collect every plane wave with ``|k+G|^2/2 <= ecut``.

        Parameters
        ----------
        cell
            Direct lattice vectors as rows, in Bohr.
        ecut
            Cutoff in Hartree.
        kpoint
            k-point in fractional coordinates of the reciprocal cell.
        """
        if ecut <= 0:
            raise ValueError(f"ecut must be positive, got {ecut}")
        cell = np.asarray(cell, dtype=float)
        reciprocal = reciprocal_cell(cell)
        kpoint = np.asarray(kpoint, dtype=float)
        k_cartesian = kpoint @ reciprocal

        gmax = np.sqrt(2.0 * ecut)
        # The box must enclose |k+G| <= gmax, i.e. |G| <= gmax + |k|.
        bounds = miller_bounds(cell, gmax + float(np.linalg.norm(k_cartesian)))
        ranges = [np.arange(-b, b + 1) for b in bounds]
        miller = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)

        cartesian = miller @ reciprocal + k_cartesian
        g2 = np.einsum("ij,ij->i", cartesian, cartesian)
        inside = g2 <= 2.0 * ecut + 1e-12
        miller, cartesian, g2 = miller[inside], cartesian[inside], g2[inside]

        if sort:
            # Lexicographic tie-break on the Miller indices makes the order deterministic
            # for the many vectors that share a shell.
            order = np.lexsort((miller[:, 2], miller[:, 1], miller[:, 0], g2))
            miller, cartesian, g2 = miller[order], cartesian[order], g2[order]

        return cls(
            miller=miller,
            cartesian=cartesian,
            g2=g2,
            kpoint=kpoint,
            ecut=float(ecut),
            cell=cell,
        )
