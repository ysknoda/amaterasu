"""Converting a Kresse-Joubert dataset into the Blochl form the rest of the code uses.

The two formulations of PAW describe the same physics but split it differently. Blochl
(Phys. Rev. B 50, 17953 (1994)) carries a *zero potential* ``v_bar``, a smooth localised
potential whose contributions cancel between the smooth and one-centre terms.
Kresse and Joubert (Phys. Rev. B 59, 1758 (1999)) never write ``v_bar`` down; they fold the
same information into a local ionic pseudopotential ``v_H[n~_Zc]``, the electrostatic
potential of the smooth core plus a compensating nuclear charge. A pslibrary ``*-kjpaw``
UPF file therefore ships the second and not the first.

The transformation between them is given in the PAW-XML specification
(https://gpaw.readthedocs.io/setups/pawxml.html, "The Kresse-Joubert formulation"):

    v_H[n~_Zc] = v_H[n~_c + (N_c - Z - N~_c) g_00 Y_00] + v_bar
                 + v_xc[n~_v + n~_c]
                 - v_xc[n~_v + n~_c + (N_v - N~_v - N~_c) g_00 Y_00] ,

which this module inverts for ``v_bar``. ``N_c`` is the number of core electrons, ``N_v``
the number of valence electrons, and ``N~_c``, ``N~_v`` the charges the *smooth* core and
valence densities actually hold.

The shape function appears on the right-hand side, so the ``v_bar`` this produces belongs
to the compensation charge it was derived with. That is not a defect of the conversion --
it is what the two bookkeepings differ by -- but it does mean the pairing has to be kept:
:func:`amaterasu.hamiltonian.paw.build_paw_machinery` says so if a run mixes them.
"""

from __future__ import annotations

import numpy as np

from ..base import RadialGrid
from .radial import derivative, integrate, radial_poisson

__all__ = ["radial_xc_potential", "zero_potential_from_kresse_joubert"]

#: Y_00 = 1/sqrt(4 pi). Radial functions are stored as the coefficient of this, so a
#: physical quantity is the stored array times Y_00 and vice versa.
ROOT_FOUR_PI = float(np.sqrt(4.0 * np.pi))


def radial_xc_potential(xc, grid: RadialGrid, density: np.ndarray) -> np.ndarray:
    """``v_xc`` of a spherically symmetric density, as a ``Y_00`` coefficient.

    For a spherical density the gradient is radial, so ``sigma = (dn/dr)^2`` and the
    functional derivative is

        v_xc = df/dn - (1/r^2) d/dr [ 2 r^2 (df/dsigma) dn/dr ] ,

    the divergence written out in spherical coordinates. The alternative used inside
    :mod:`amaterasu.pseudo.paw.onsite` -- integrating the gradient term by parts -- gives
    matrix elements rather than a potential, and a potential is what is needed here.
    """
    physical = np.asarray(density, dtype=float) / ROOT_FOUR_PI
    r = grid.r
    # The functional is always called spin-resolved; an unpolarised density is half in
    # each channel, and both channels then carry the same potential.
    half = physical / 2.0
    rho = np.stack([half, half], axis=-1)
    if not xc.is_gga:
        _, vrho = xc.evaluate(rho)
        return np.asarray(vrho)[:, 0] * ROOT_FOUR_PI

    slope = derivative(grid, half)
    sigma = np.stack([slope**2, slope**2, slope**2], axis=-1)
    _, vrho, vsigma = xc.evaluate(rho, sigma)
    vrho = np.asarray(vrho)[:, 0]
    vsigma = np.asarray(vsigma)
    # For one spin channel the gradient term is
    #     -div [ 2 (df/dsigma_uu) grad n_up + (df/dsigma_ud) grad n_dn ] ,
    # and with both channels carrying grad n / 2 the bracket is this coefficient times the
    # half-channel slope already in hand.
    coefficient = 2.0 * vsigma[:, 0] + vsigma[:, 1]
    flux = coefficient * slope * r**2
    inverse = np.where(r > 0, 1.0 / np.where(r > 0, r**2, 1.0), 0.0)
    potential = vrho - derivative(grid, flux) * inverse
    # r = 0 is a coordinate singularity, not a physical one; the divergence term is finite
    # there and the neighbouring point is the closest honest value available.
    if r[0] <= 0.0:
        potential[0] = potential[1]
    return potential * ROOT_FOUR_PI


def zero_potential_from_kresse_joubert(
    xc,
    grid: RadialGrid,
    *,
    local_potential: np.ndarray,
    pseudo_core_density: np.ndarray,
    pseudo_valence_density: np.ndarray,
    shape,
    core_electrons: float,
    valence_electrons: float,
    atomic_number: float,
) -> np.ndarray:
    """``v_bar`` from ``v_H[n~_Zc]``, all arrays being ``Y_00`` coefficients in Hartree."""
    r = grid.r
    smooth_core = integrate(grid, ROOT_FOUR_PI * pseudo_core_density * r**2)
    smooth_valence = integrate(grid, ROOT_FOUR_PI * pseudo_valence_density * r**2)

    def smeared(charge: float) -> np.ndarray:
        """A spherical charge ``charge`` with the compensation shape, as a Y_00 coefficient."""
        return charge * shape.radial(0, r) / ROOT_FOUR_PI

    core_charge = core_electrons - atomic_number - smooth_core
    valence_charge = valence_electrons - smooth_valence - smooth_core

    hartree = radial_poisson(grid, 0, pseudo_core_density + smeared(core_charge))
    smooth = pseudo_valence_density + pseudo_core_density
    return (
        local_potential
        - hartree
        - radial_xc_potential(xc, grid, smooth)
        + radial_xc_potential(xc, grid, smooth + smeared(valence_charge))
    )
