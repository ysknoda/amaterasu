"""Radial Fourier transforms: from a tabulated pseudopotential to plane-wave form factors.

Everything a norm-conserving pseudopotential contributes to a plane-wave Hamiltonian is a
spherically symmetric radial function, so each one enters through a single
spherical-Bessel transform

    f~(q) = integral_0^inf f(r) j_l(q r) dr,

where ``f`` already carries whatever powers of ``r`` the stored convention supplies
(see :mod:`amaterasu.pseudo.base`). The three users are the local potential (``l = 0``),
the Kleinman-Bylander projectors (``l`` of the channel), and the atomic densities that
seed the SCF (``l = 0``).

The Coulomb tail
----------------
The local potential is the one transform that cannot be taken as it stands: ``V_loc(r)``
decays as ``-Z_ion/r``, whose transform ``-4 pi Z / q^2`` diverges at ``q = 0`` and whose
radial integral does not converge on a truncated mesh. The standard cure (Quantum
ESPRESSO's ``vloc_of_g``, ABINIT's ``psp_ ...``) is to split off an *analytic* Coulomb
piece screened by an error function,

    V_loc(r) = [V_loc(r) + Z erf(r)/r] - Z erf(r)/r,

whose bracketed remainder decays like ``exp(-r^2)`` and transforms cleanly, while the
subtracted part has the closed form ``-4 pi Z exp(-q^2/4) / q^2``. What is left at
``q = 0`` after the ``-4 pi Z / q^2`` divergence is set aside -- to be cancelled against
the Hartree and Ewald ``G = 0`` terms -- is the finite "alpha" integral

    v~_loc(0) = 4 pi integral r [r V_loc(r) + Z] dr,

which is what :func:`local_potential_form_factor` returns for ``q = 0``. Both routes to
that number agree analytically because ``integral_0^inf r erfc(r) dr = 1/4``; the test
suite checks the ``q -> 0`` limit numerically as well.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf, spherical_jn

from .base import NCPseudopotential, Projector, RadialGrid

__all__ = [
    "density_form_factor",
    "density_form_factor_derivative",
    "local_potential_form_factor",
    "local_potential_form_factor_derivative",
    "projector_form_factor",
    "projector_form_factor_derivative",
    "radial_transform",
    "radial_transform_derivative",
]


def radial_transform(grid: RadialGrid, values: np.ndarray, l: int, q: np.ndarray) -> np.ndarray:
    """``integral f(r) j_l(q r) dr`` evaluated at every ``q``.

    ``values`` must already include any powers of ``r`` demanded by the stored convention:
    for a density stored as ``4 pi r^2 rho`` this returns the true Fourier transform
    ``integral rho(r) exp(-i q.r) d^3r``, and for a projector stored as ``r beta(r)`` the
    caller passes ``r * values`` to get ``integral r^2 beta j_l dr``.

    Evaluated directly rather than through an interpolation table: the transform depends
    only on ``|q|``, so it is computed once per geometry and reused across the SCF loop.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    values = np.asarray(values, dtype=float)
    if values.shape != grid.r.shape:
        raise ValueError(f"values has shape {values.shape}, grid has {grid.r.shape}")
    integrand = values * spherical_jn(int(l), q[:, None] * grid.r[None, :])
    return grid.integrate(integrand)


def local_potential_form_factor(pseudo: NCPseudopotential, q: np.ndarray) -> np.ndarray:
    """``v~_loc(q)`` in Hartree*Bohr^3, regularised at ``q = 0``.

    For ``q > 0`` this is the complete transform, Coulomb tail included -- that tail is
    the physical electron-nucleus attraction and must stay. Only at ``q = 0`` is the
    ``-4 pi Z / q^2`` divergence set aside, to be cancelled against the matching Hartree
    and Ewald ``G = 0`` divergences of a charge-neutral cell; what is returned there is
    the finite remainder ``4 pi integral r [r V_loc + Z] dr``.

    The two branches join up: ``v~_loc(q) + 4 pi Z / q^2 -> v~_loc(0)`` as ``q -> 0``,
    because ``integral_0^inf r erfc(r) dr = 1/4`` exactly accounts for the difference
    between the screened and unscreened subtractions.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    grid, z = pseudo.grid, float(pseudo.valence_charge)
    r = grid.r
    result = np.empty_like(q)

    at_zero = q < 1e-12
    if np.any(at_zero):
        # 4 pi integral r [r V_loc(r) + Z] dr -- finite because r V_loc -> -Z.
        result[at_zero] = 4.0 * np.pi * grid.integrate(r * (r * pseudo.local_potential + z))

    if np.any(~at_zero):
        qq = q[~at_zero]
        screened = r * pseudo.local_potential + z * erf(r)
        integrand = screened * np.sin(qq[:, None] * r[None, :])
        short_range = 4.0 * np.pi / qq * grid.integrate(integrand)
        result[~at_zero] = short_range - 4.0 * np.pi * z * np.exp(-0.25 * qq**2) / qq**2
    return result


def projector_form_factor(grid: RadialGrid, projector: Projector, q: np.ndarray) -> np.ndarray:
    """``f_l(q) = integral r^2 beta_l(r) j_l(q r) dr`` in Bohr^(3/2).

    The stored projector is ``r beta_l(r)``, so the integrand is ``r * values``. With the
    generators' normalisation ``integral (r beta)^2 dr = 1``, the Hankel-transform
    identity gives ``(2/pi) integral q^2 f_l(q)^2 dq = 1``, which is the completeness sum
    rule the plane-wave projector must approach as the cutoff grows.
    """
    return radial_transform(grid, grid.r * projector.values, projector.l, q)


def density_form_factor(grid: RadialGrid, density: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``rho~(q) = integral rho(r) exp(-i q.r) d^3r`` for a density stored as ``4 pi r^2 rho``.

    At ``q = 0`` this is the enclosed charge, so it is also the cheapest check that a
    density was read with the right radial convention.
    """
    return radial_transform(grid, density, 0, q)


# --------------------------------------------------------------------------- derivatives
# The stress tensor needs d/dq of every form factor, because straining the cell stretches
# each |k+G|. These are the exact radial derivatives, obtained by differentiating the
# spherical Bessel function under the integral -- not by finite differences of the
# transforms above, which would inherit their quadrature error.


def radial_transform_derivative(
    grid: RadialGrid, values: np.ndarray, l: int, q: np.ndarray
) -> np.ndarray:
    """``d/dq integral f(r) j_l(q r) dr = integral f(r) r j_l'(q r) dr``."""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    values = np.asarray(values, dtype=float)
    if values.shape != grid.r.shape:
        raise ValueError(f"values has shape {values.shape}, grid has {grid.r.shape}")
    argument = q[:, None] * grid.r[None, :]
    integrand = values * grid.r * spherical_jn(int(l), argument, derivative=True)
    return grid.integrate(integrand)


def density_form_factor_derivative(
    grid: RadialGrid, density: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """``d rho~/dq`` for a density stored as ``4 pi r^2 rho``."""
    return radial_transform_derivative(grid, density, 0, q)


def projector_form_factor_derivative(
    grid: RadialGrid, projector: Projector, q: np.ndarray
) -> np.ndarray:
    """``d f_l/dq`` for the Kleinman-Bylander projector."""
    return radial_transform_derivative(grid, grid.r * projector.values, projector.l, q)


def local_potential_form_factor_derivative(pseudo: NCPseudopotential, q: np.ndarray) -> np.ndarray:
    """``d v~_loc/dq``, differentiating both pieces of the error-function split.

    ``v~_loc`` depends on ``q`` only through ``q^2``, so the derivative vanishes at the
    origin; it is set to zero there rather than evaluated, since both terms are
    individually singular and only their sum is finite.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    grid, z = pseudo.grid, float(pseudo.valence_charge)
    r = grid.r
    result = np.zeros_like(q)

    nonzero = q >= 1e-12
    if not np.any(nonzero):
        return result
    qq = q[nonzero]
    screened = r * pseudo.local_potential + z * erf(r)
    argument = qq[:, None] * r[None, :]
    integral = grid.integrate(screened * np.sin(argument))
    derivative = grid.integrate(screened * r * np.cos(argument))
    result[nonzero] = (
        -4.0 * np.pi / qq**2 * integral
        + 4.0 * np.pi / qq * derivative
        + 4.0 * np.pi * z * np.exp(-0.25 * qq**2) * (0.5 / qq + 2.0 / qq**3)
    )
    return result
