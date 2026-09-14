"""Ewald summation for the electrostatic energy of point ions in a periodic cell.

The bare ion-ion sum ``(1/2) sum'_{R,ab} Z_a Z_b / |tau_ab + R|`` is only conditionally
convergent. Ewald's device is to split ``1/r`` with an error function into a short-ranged
part summed in real space and a smooth part summed in reciprocal space:

    E = (1/2) sum'_{R,ab} Z_a Z_b erfc(eta r) / r
      + (2 pi / Omega) sum_{G != 0} |S(G)|^2 exp(-G^2 / 4 eta^2) / G^2
      - (eta / sqrt(pi)) sum_a Z_a^2
      - (pi / (2 Omega eta^2)) (sum_a Z_a)^2 ,

with ``S(G) = sum_a Z_a exp(i G.tau_a)``. The third term removes each Gaussian's
interaction with itself, which the reciprocal sum includes. The fourth is the ``G = 0``
term: dropping it alone would be wrong, since the divergences of the ion-ion, Hartree and
electron-ion ``G = 0`` terms cancel only when taken together. For a neutral cell the
three divergent pieces are ``2 pi Z^2 / (Omega G^2)``, ``2 pi N_e^2 / (Omega G^2)`` and
``-4 pi Z N_e / (Omega G^2)``, summing to ``2 pi (N_e - Z)^2 / (Omega G^2) = 0``; the
finite remainders are the term above, nothing (Hartree), and the alpha integral in
:mod:`amaterasu.pseudo.formfactors`.

The splitting parameter ``eta`` is a pure convenience: the total is independent of it, and
:func:`ewald_energy` is tested across a wide range of ``eta`` to prove it. Its default
balances the cost of the two sums, giving comparable term counts on either side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

from ..basis.gvectors import miller_bounds, reciprocal_cell

__all__ = ["EwaldResult", "ewald", "ewald_energy", "optimal_eta"]


@dataclass
class EwaldResult:
    """Ion-ion energy in Hartree, forces in Ha/Bohr, and stress in Ha/Bohr^3.

    The stress follows the convention used throughout amaterasu (and by ASE),
    ``sigma_ab = (1/Omega) dE/d eps_ab``, so a positive hydrostatic pressure
    ``P = -tr(sigma)/3`` is one that would expand the cell.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray


def optimal_eta(volume: float) -> float:
    """Splitting parameter that balances the real- and reciprocal-space term counts.

    Matching the two sphere volumes -- ``(4 pi/3) r_c^3 / Omega`` lattice points against
    ``(4 pi/3) g_c^3 Omega / (2 pi)^3`` reciprocal points, with ``r_c ~ 5.5/eta`` and
    ``g_c ~ 11 eta`` for a 1e-14 tail -- gives ``eta ~ 1.8 / Omega^(1/3)``.
    """
    return 1.8 / volume ** (1.0 / 3.0)


def _real_space_cutoff(eta: float, tolerance: float) -> float:
    """Radius beyond which ``erfc(eta r)/r`` is below ``tolerance``."""
    radius = 1.0 / eta
    for _ in range(200):
        if erfc(eta * radius) / radius < tolerance:
            return radius
        radius *= 1.2
    raise RuntimeError("the Ewald real-space sum failed to converge")  # pragma: no cover


def _reciprocal_cutoff(eta: float, volume: float, charge2: float, tolerance: float) -> float:
    """Radius beyond which the reciprocal-space summand is below ``tolerance``."""
    g = 2.0 * eta
    for _ in range(200):
        if 2.0 * np.pi / volume * charge2 * np.exp(-(g**2) / (4.0 * eta**2)) / g**2 < tolerance:
            return g
        g *= 1.2
    raise RuntimeError("the Ewald reciprocal-space sum failed to converge")  # pragma: no cover


def ewald_energy(
    cell: np.ndarray,
    positions: np.ndarray,
    charges: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1e-14,
) -> float:
    """Electrostatic energy of point charges in a periodic cell, in Hartree."""
    return ewald(cell, positions, charges, eta=eta, tolerance=tolerance).energy


def ewald(
    cell: np.ndarray,
    positions: np.ndarray,
    charges: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1e-14,
) -> EwaldResult:
    """Energy, forces and stress of point charges in a periodic cell.

    Parameters
    ----------
    cell
        Lattice vectors as rows, in Bohr.
    positions
        Cartesian positions in Bohr, shape ``(natoms, 3)``.
    charges
        Ionic charges (``Z_ion``, positive), shape ``(natoms,)``.
    eta
        Splitting parameter in 1/Bohr; ``None`` uses :func:`optimal_eta`. No result
        depends on it.
    tolerance
        Size below which a term in either sum is dropped.

    Notes
    -----
    Both derivatives follow from differentiating the same two sums. In real space a pair
    separation strains as ``r -> r + eps r``, so ``dr/d eps_uv = r_u r_v / r``; in
    reciprocal space ``G`` strains the other way, ``dG^2/d eps_uv = -2 G_u G_v``, while
    the structure factor ``S(G) = sum_a Z_a exp(i G.tau_a)`` is invariant because
    ``G.tau`` is. The ``-pi Z^2 / (2 Omega eta^2)`` background term scales as ``1/Omega``
    and so contributes ``-delta_uv`` times itself; the Gaussian self-energy does not
    depend on the geometry at all and contributes nothing.
    """
    cell = np.asarray(cell, dtype=float)
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    charges = np.atleast_1d(np.asarray(charges, dtype=float))
    if len(positions) != len(charges):
        raise ValueError(f"{len(positions)} positions but {len(charges)} charges")
    volume = abs(float(np.linalg.det(cell)))
    if eta is None:
        eta = optimal_eta(volume)
    total_charge = float(charges.sum())
    natoms = len(charges)

    forces = np.zeros((natoms, 3))
    stress = np.zeros((3, 3))

    # ---------------------------------------------------------------- real space
    # Fold each pair separation into the cell first, so the translation box only has to
    # cover the cutoff sphere and not the cell diagonal on top of it.
    inverse = np.linalg.inv(cell)
    separations = positions[:, None, :] - positions[None, :, :]
    fractional = separations @ inverse
    fractional -= np.rint(fractional)
    separations = fractional @ cell

    reciprocal = reciprocal_cell(cell)
    r_cut = _real_space_cutoff(eta, tolerance / max(1.0, total_charge**2))
    # An extra shell absorbs the folding above, which can move a pair by half a cell.
    repeats = np.ceil(r_cut * np.linalg.norm(reciprocal, axis=1) / (2.0 * np.pi)).astype(int) + 1
    offsets = (
        np.stack(
            np.meshgrid(*[np.arange(-n, n + 1) for n in repeats], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        @ cell
    )

    charge_products = charges[:, None] * charges[None, :]
    real_sum = 0.0
    for offset in offsets:
        vectors = separations + offset
        distances = np.linalg.norm(vectors, axis=-1)
        # The self term (same atom, zero translation) is excluded from the lattice sum;
        # its Gaussian self-interaction is removed analytically below instead.
        near_zero = distances < 1e-10
        safe = np.where(near_zero, 1.0, distances)
        pair = np.where(near_zero, 0.0, charge_products * erfc(eta * safe) / safe)
        if np.max(np.abs(pair)) < tolerance and np.min(safe) > r_cut:
            continue
        real_sum += float(pair.sum())

        # d/dr [erfc(eta r)/r] = -erfc(eta r)/r^2 - (2 eta/sqrt(pi)) exp(-eta^2 r^2)/r
        derivative = np.where(
            near_zero,
            0.0,
            -charge_products
            * (
                erfc(eta * safe) / safe**2
                + 2.0 * eta / np.sqrt(np.pi) * np.exp(-((eta * safe) ** 2)) / safe
            ),
        )
        # The 1/2 of the double sum cancels against tau_a appearing in both slots.
        forces -= np.einsum("ab,abi,ab->ai", derivative, vectors, 1.0 / safe)
        stress += 0.5 * np.einsum("ab,abu,abv,ab->uv", derivative, vectors, vectors, 1.0 / safe)
    real_sum *= 0.5

    # ---------------------------------------------------------- reciprocal space
    g_cut = _reciprocal_cutoff(eta, volume, max(total_charge**2, 1.0), tolerance)
    bounds = miller_bounds(cell, g_cut)
    miller = np.stack(
        np.meshgrid(*[np.arange(-n, n + 1) for n in bounds], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    gvectors = miller @ reciprocal
    g2 = np.einsum("ij,ij->i", gvectors, gvectors)
    keep = (g2 > 1e-12) & (g2 <= g_cut**2)
    gvectors, g2 = gvectors[keep], g2[keep]
    phases = np.exp(1j * (gvectors @ positions.T))
    structure = phases @ charges
    weight = np.exp(-g2 / (4.0 * eta**2)) / g2
    reciprocal_sum = float(2.0 * np.pi / volume * np.sum(np.abs(structure) ** 2 * weight))

    # d|S|^2/d tau_a = -2 Z_a G Im(S* exp(i G.tau_a)), and the force is minus that.
    imaginary = np.imag(structure.conj()[:, None] * phases)
    forces += (
        4.0
        * np.pi
        / volume
        * charges[:, None]
        * np.einsum("g,ga,gi->ai", weight, imaginary, gvectors)
    )

    factor = np.abs(structure) ** 2 * weight * (1.0 / (2.0 * eta**2) + 2.0 / g2)
    stress += 2.0 * np.pi / volume * np.einsum("g,gu,gv->uv", factor, gvectors, gvectors)
    stress -= np.eye(3) * reciprocal_sum

    self_energy = -eta / np.sqrt(np.pi) * float(np.sum(charges**2))
    background = -np.pi / (2.0 * volume * eta**2) * total_charge**2
    stress -= np.eye(3) * background

    return EwaldResult(
        energy=real_sum + reciprocal_sum + self_energy + background,
        forces=forces,
        stress=stress / volume,
    )
