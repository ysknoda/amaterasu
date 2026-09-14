"""Local ionic potential and the atomic densities that seed an SCF calculation.

Everything here is built in reciprocal space on the density sphere and transformed once
to the grid. A quantity that is a superposition of spherical atomic functions,
``F(r) = sum_a f_a(|r - tau_a|)``, has the Fourier coefficients

    F(G) = (1/Omega) sum_a f~_a(|G|) exp(-i G.tau_a),

so the radial transform is done once per species (it depends only on ``|G|``) and combined
with a per-atom phase. The ``1/Omega`` is the convention of :mod:`amaterasu.basis.fft`,
where ``F(r) = sum_G F(G) exp(i G.r)`` with no volume factor in the transform.

The ``G = 0`` component of the local potential is the finite alpha term described in
:mod:`amaterasu.pseudo.formfactors`; it is a constant shift of the potential that pairs
with the ``G = 0`` terms of the Hartree and Ewald energies, and dropping it silently
would move the total energy by several eV per atom.
"""

from __future__ import annotations

import numpy as np

from ..basis.fft import FFTGrid
from ..basis.gvectors import GSphere
from ..pseudo.base import NCPseudopotential
from ..pseudo.formfactors import density_form_factor, local_potential_form_factor

__all__ = [
    "atomic_core_density",
    "atomic_valence_density",
    "core_density_ringing",
    "local_potential",
    "structure_factors",
]


def structure_factors(sphere: GSphere, positions: np.ndarray) -> np.ndarray:
    """``exp(-i G.tau_a)`` for every atom and plane wave, shape ``(natoms, npw)``."""
    return np.exp(-1j * (np.atleast_2d(positions) @ sphere.cartesian.T))


def _superpose(
    grid: FFTGrid,
    sphere: GSphere,
    positions: np.ndarray,
    symbols: list[str],
    radial: dict[str, np.ndarray],
) -> np.ndarray:
    """Sum per-species radial form factors with atomic phases and transform to real space."""
    phases = structure_factors(sphere, positions)
    coefficients = np.zeros(len(sphere), dtype=complex)
    for atom, symbol in enumerate(symbols):
        coefficients += radial[symbol] * phases[atom]
    coefficients /= grid.volume
    return grid.to_real(grid.scatter(coefficients, sphere))


def local_potential(
    grid: FFTGrid,
    sphere: GSphere,
    positions: np.ndarray,
    symbols: list[str],
    pseudopotentials: dict[str, NCPseudopotential],
) -> np.ndarray:
    """Local pseudopotential on the real-space grid, in Hartree.

    ``positions`` are Cartesian in Bohr and ``sphere`` is the density sphere at Gamma.
    """
    q = np.sqrt(sphere.g2)
    radial = {
        symbol: local_potential_form_factor(pseudo, q)
        for symbol, pseudo in pseudopotentials.items()
    }
    return _real(_superpose(grid, sphere, positions, symbols, radial), "local potential")


def atomic_valence_density(
    grid: FFTGrid,
    sphere: GSphere,
    positions: np.ndarray,
    symbols: list[str],
    pseudopotentials: dict[str, NCPseudopotential],
    *,
    total_electrons: float | None = None,
) -> np.ndarray:
    """Superposition of atomic valence pseudo-densities, in electrons/Bohr^3.

    This is the SCF starting guess. It is rescaled to hold exactly ``total_electrons``
    (the pseudo-atom densities integrate slightly short of ``Z_ion`` because the radial
    mesh is truncated), and clipped at a small positive floor: the truncated
    superposition can dip marginally below zero in the interstitial region, which a GGA
    would then evaluate at an unphysical density.
    """
    q = np.sqrt(sphere.g2)
    radial = {}
    for symbol, pseudo in pseudopotentials.items():
        if pseudo.valence_density is None:
            raise ValueError(f"{symbol}: the dataset carries no atomic valence density")
        radial[symbol] = density_form_factor(pseudo.grid, pseudo.valence_density, q)
    density = _real(_superpose(grid, sphere, positions, symbols, radial), "valence density")

    density = np.maximum(density, 1e-12)
    if total_electrons is not None:
        current = float(np.sum(density)) * grid.cell_volume_per_point
        density *= total_electrons / current
    return density


def atomic_core_density(
    grid: FFTGrid,
    sphere: GSphere,
    positions: np.ndarray,
    symbols: list[str],
    pseudopotentials: dict[str, NCPseudopotential],
) -> np.ndarray | None:
    """Partial core density for the nonlinear core correction, or ``None`` if unused.

    The NLCC (Louie, Froyen, Cohen, PRB 26, 1738 (1982)) adds a frozen pseudised core
    density inside the exchange-correlation functional only -- never to the Hartree term
    or the electron count -- so that ``E_xc`` sees a total density closer to the
    all-electron one. Every PseudoDojo dataset carries it, and leaving it out changes
    lattice constants at the 0.01 Angstrom level, so it is applied whenever present.

    The result is **not** clipped at zero, even though the Fourier synthesis of a sharply
    peaked core rings slightly negative in the interstitial. Clipping would be a
    non-smooth function of the atomic positions, and the analytic force through ``E_xc``
    -- which assumes the core is a rigid function that merely translates -- would then be
    wrong: measured on a badly under-converged oxygen cell, clipping put the NLCC force a
    factor of five out. The ringing is a symptom of too small a density cutoff and shrinks
    with it (for the PseudoDojo oxygen dataset, from 3% of the peak at 10 Ha to 6e-7 of it
    at the recommended 42 Ha), so the honest response is to report it, not to hide it;
    :func:`core_density_ringing` measures it and the SCF driver warns when it is large.
    """
    if not any(p.has_core_correction for p in pseudopotentials.values()):
        return None
    q = np.sqrt(sphere.g2)
    radial = {}
    for symbol, pseudo in pseudopotentials.items():
        if pseudo.core_density is None:
            radial[symbol] = np.zeros_like(q)
        else:
            radial[symbol] = density_form_factor(pseudo.grid, pseudo.core_density, q)
    return _real(_superpose(grid, sphere, positions, symbols, radial), "core density")


def core_density_ringing(core_density: np.ndarray | None) -> float:
    """How far the synthesised core density dips below zero, relative to its peak.

    A rigid, positive atomic function represented on a finite G-sphere overshoots; the
    size of the overshoot is a direct measure of how well the density cutoff resolves the
    core. Anything above about 1e-3 means the cutoff is too small for this dataset.
    """
    if core_density is None:
        return 0.0
    peak = float(np.max(core_density))
    return 0.0 if peak <= 0 else float(-np.min(core_density)) / peak


def _real(field: np.ndarray, what: str) -> np.ndarray:
    """Take the real part of a field that must be real, checking that it is."""
    scale = float(np.max(np.abs(field.real)))
    residual = float(np.max(np.abs(field.imag)))
    if scale > 0 and residual > 1e-8 * scale:
        raise ValueError(
            f"the {what} came out complex (imaginary part {residual:.3e} against "
            f"{scale:.3e}); the sphere-to-grid mapping or a phase convention is wrong"
        )
    return np.ascontiguousarray(field.real)
