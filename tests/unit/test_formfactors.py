"""Radial Fourier transforms of the pseudopotential.

Each check here is an identity the transform must satisfy, not a comparison against a
stored number, so they stay valid for any dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.pseudo.formfactors import (
    density_form_factor,
    local_potential_form_factor,
    projector_form_factor,
    radial_transform,
)


def test_density_form_factor_at_zero_is_the_enclosed_charge(si_pseudo):
    """rho~(0) = integral rho d^3r, so it must equal the integrated valence charge."""
    value = density_form_factor(si_pseudo.grid, si_pseudo.valence_density, np.array([0.0]))[0]
    assert value == pytest.approx(si_pseudo.integrated_valence_charge(), rel=1e-12)
    # ... which is Z_ion up to the truncation of the radial mesh.
    assert value == pytest.approx(si_pseudo.valence_charge, rel=0.02)


def test_density_form_factor_decays(si_pseudo):
    """A smooth density has a form factor that falls monotonically at small q."""
    q = np.linspace(0.0, 4.0, 40)
    values = density_form_factor(si_pseudo.grid, si_pseudo.valence_density, q)
    assert np.all(np.diff(values[:20]) < 0)
    assert abs(values[-1]) < 0.1 * abs(values[0])


def test_local_potential_branches_join_at_q_zero(si_pseudo):
    """v~(q) + 4 pi Z / q^2 -> v~(0): the two code paths must agree in the limit.

    The screened and unscreened subtractions differ by exactly ``4 pi Z * integral r
    erfc(r) dr = pi Z``, so this catches a wrong prefactor in either branch.
    """
    z = si_pseudo.valence_charge
    at_zero = local_potential_form_factor(si_pseudo, np.array([0.0]))[0]
    q = np.array([1e-2, 3e-3, 1e-3])
    limit = local_potential_form_factor(si_pseudo, q) + 4.0 * np.pi * z / q**2
    assert limit[-1] == pytest.approx(at_zero, rel=1e-6)
    # Approaching quadratically, so a 10x smaller q is 100x closer.
    errors = np.abs(limit - at_zero)
    assert errors[-1] < errors[0]


def test_local_potential_is_coulombic_at_small_q_and_soft_at_large_q(si_pseudo):
    """The long-range tail is Coulombic; the short-range core is not.

    At small ``q`` the transform samples the ``-Z/r`` tail and must approach
    ``-4 pi Z / q^2``. At large ``q`` it must fall away *faster* than that, because a
    pseudopotential is finite at the origin -- it has no ``1/r`` singularity to produce a
    ``1/q^2`` transform. Softness at the origin is exactly what makes the plane-wave
    basis converge, so this is the numerical statement of "pseudo".
    """
    z = si_pseudo.valence_charge
    small = np.array([0.05, 0.1])
    coulomb = -4.0 * np.pi * z / small**2
    assert local_potential_form_factor(si_pseudo, small) == pytest.approx(coulomb, rel=0.01)

    large = np.array([30.0, 45.0])
    value = local_potential_form_factor(si_pseudo, large)
    assert np.all(np.abs(value) < 1e-3 * np.abs(-4.0 * np.pi * z / large**2))


@pytest.mark.parametrize("index", range(6))
def test_projector_completeness_sum_rule(si_pseudo, index):
    """(2/pi) integral q^2 f_l(q)^2 dq = 1 follows from the generators' normalisation.

    This is the Hankel-transform statement of ``integral (r beta)^2 dr = 1``, so it tests
    the radial convention, the Bessel order and the quadrature together.
    """
    projector = si_pseudo.projectors[index]
    q = np.linspace(0.0, 80.0, 6001)
    f = projector_form_factor(si_pseudo.grid, projector, q)
    assert 2.0 / np.pi * np.trapezoid(q**2 * f**2, q) == pytest.approx(1.0, abs=2e-5)


def test_projector_form_factor_vanishes_at_q_zero_for_l_above_zero(si_pseudo):
    """j_l(0) = 0 for l > 0, so only s projectors survive at the origin."""
    q = np.array([0.0])
    for projector in si_pseudo.projectors:
        value = projector_form_factor(si_pseudo.grid, projector, q)[0]
        if projector.l == 0:
            assert abs(value) > 1e-3
        else:
            assert value == pytest.approx(0.0, abs=1e-14)


def test_radial_transform_matches_an_analytic_gaussian():
    """For f(r) = 4 pi r^2 exp(-a r^2) the l=0 transform is known in closed form."""
    from amaterasu.pseudo.base import RadialGrid

    a = 1.7
    r = np.linspace(0.0, 12.0, 4001)
    grid = RadialGrid(r=r)
    values = 4.0 * np.pi * r**2 * np.exp(-a * r**2)
    q = np.array([0.0, 0.5, 1.0, 2.5, 5.0])
    # integral exp(-a r^2) exp(-i q.r) d^3r = (pi/a)^(3/2) exp(-q^2 / 4a)
    expected = (np.pi / a) ** 1.5 * np.exp(-(q**2) / (4.0 * a))
    assert radial_transform(grid, values, 0, q) == pytest.approx(expected, rel=1e-10)


def test_radial_transform_rejects_a_shape_mismatch(si_pseudo):
    with pytest.raises(ValueError, match="shape"):
        radial_transform(si_pseudo.grid, np.zeros(3), 0, np.array([1.0]))
