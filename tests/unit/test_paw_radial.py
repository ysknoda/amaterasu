"""Radial quadrature, differentiation and electrostatics inside the augmentation sphere."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

from amaterasu.pseudo.base import RadialGrid
from amaterasu.pseudo.paw.radial import (
    cumulative_integral,
    derivative,
    integrate,
    multipole_moment,
    poisson_operator,
    radial_hartree_energy,
    radial_poisson,
)

#: The grid parameters of a real PAW-XML dataset (JTH silicon), so the tests exercise the
#: mesh the code actually meets rather than a convenient uniform one.
A, D, N = 1.5534785893308882e-03, 2.1691728751247669e-02, 500


def log_grid(n=N, d=D) -> RadialGrid:
    index = np.arange(n, dtype=float)
    return RadialGrid(r=A * (np.exp(d * index) - 1.0), kind="log", dr_di=A * d * np.exp(d * index))


def test_log_mesh_integration_is_fourth_order():
    """Integrating in r rather than in the index would be second order; check it is not."""
    coarse, fine = log_grid(), log_grid(2 * N, D / 2)
    exact = np.sqrt(np.pi) / 4 / 2.5**1.5
    values = [integrate(g, np.exp(-2.5 * g.r**2) * g.r**2) - exact for g in (coarse, fine)]
    assert abs(values[0]) < 1e-12


def test_cumulative_integral_is_exact_for_cubics():
    size = 12
    grid = RadialGrid(r=np.arange(size, dtype=float), kind="linear", dr_di=np.ones(size))
    x = grid.r
    values = 3 - 2 * x + 0.5 * x**2 - 0.1 * x**3
    exact = 3 * x - x**2 + x**3 / 6 - 0.025 * x**4
    assert cumulative_integral(grid, values) == pytest.approx(exact, abs=1e-12)


def test_derivative_is_exact_for_quartics():
    size = 12
    grid = RadialGrid(r=np.arange(size, dtype=float), kind="linear", dr_di=np.ones(size))
    x = grid.r
    values = 3 - 2 * x + 0.5 * x**2 - 0.1 * x**3 + 0.02 * x**4
    exact = -2 + x - 0.3 * x**2 + 0.08 * x**3
    assert derivative(grid, values) == pytest.approx(exact, abs=1e-11)


def test_derivative_handles_a_truncated_array():
    grid = log_grid()
    full = derivative(grid, np.exp(-grid.r))
    part = derivative(grid, np.exp(-grid.r[:100]))
    assert part[50] == pytest.approx(full[50], rel=1e-9)


def test_poisson_reproduces_the_gaussian_potential():
    """A Gaussian charge has the closed-form potential Q erf(sqrt(a) r)/r."""
    grid = log_grid()
    r, alpha = grid.r, 2.5
    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    density = np.exp(-alpha * r**2)
    charge = np.sqrt(4.0 * np.pi) * integrate(grid, density * r**2)
    potential = radial_poisson(grid, 0, density)
    exact = np.empty_like(r)
    exact[1:] = charge * erf(np.sqrt(alpha) * r[1:]) / r[1:] / y00
    exact[0] = charge * 2.0 * np.sqrt(alpha / np.pi) / y00
    assert potential == pytest.approx(exact, abs=1e-6)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_poisson_higher_multipoles_against_direct_quadrature(l):
    grid = log_grid()
    r, alpha = grid.r, 2.5
    density = r**l * np.exp(-alpha * r**2)
    potential = radial_poisson(grid, l, density)
    for target in (0.3, 1.0, 2.5):
        j = int(np.argmin(np.abs(r - target)))
        radius = r[j]
        inner = quad(lambda s: s ** (2 * l + 2) * np.exp(-alpha * s**2), 0, radius)[0]
        outer = quad(lambda s: s * np.exp(-alpha * s**2), radius, np.inf)[0]
        reference = 4.0 * np.pi / (2 * l + 1) * (inner / radius ** (l + 1) + radius**l * outer)
        assert potential[j] == pytest.approx(reference, rel=1e-5)


def test_gaussian_self_energy():
    """The Coulomb self-energy of a unit Gaussian charge is Q^2 sqrt(2 alpha / pi) / 2."""
    grid, alpha = log_grid(), 2.5
    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    density = (alpha / np.pi) ** 1.5 * np.exp(-alpha * grid.r**2) / y00
    assert multipole_moment(grid, 0, density) * np.sqrt(4 * np.pi) == pytest.approx(1.0, abs=1e-12)
    exact = 0.5 * np.sqrt(2 * alpha / np.pi)
    assert radial_hartree_energy(grid, {0: density}) == pytest.approx(exact, abs=1e-7)


def test_poisson_operator_is_symmetric_and_preserves_the_energy():
    """The whole point of the dense operator: the same energy, an exactly self-adjoint form.

    ``radial_poisson`` is accurate but not self-adjoint on the mesh, because the running
    integral over the source and the outer integral over the field point use different
    quadrature weights. That leaves the energy right and its gradient several per cent
    wrong -- an error no energy check can see.
    """
    grid, stop, alpha = log_grid(), 300, 2.5
    r = grid.r[:stop]
    weights = integrate(grid, np.eye(stop), upto=stop)
    measure = weights * r**2
    operator = poisson_operator(grid, 0, upto=stop, measure=measure)

    matrix = measure[:, None] * operator
    assert np.abs(matrix - matrix.T).max() < 1e-12 * np.abs(matrix).max()

    density = np.exp(-alpha * r**2)
    symmetric = 0.5 * float(np.sum(measure * density * (operator @ density)))
    split = 0.5 * float(
        np.sum(measure * density * radial_poisson(grid, 0, density, upto=stop)[:stop])
    )
    assert symmetric == pytest.approx(split, rel=1e-13)


def test_poisson_operator_gradient_is_exact():
    """d/dn of (1/2) n.W.v[n] must be exactly W v; that is what the on-site Hamiltonian uses."""
    grid, stop = log_grid(), 200
    r = grid.r[:stop]
    weights = integrate(grid, np.eye(stop), upto=stop)
    measure = weights * r**2
    operator = poisson_operator(grid, 1, upto=stop, measure=measure)

    def energy(density):
        return 0.5 * float(np.sum(measure * density * (operator @ density)))

    density = r * np.exp(-1.7 * r**2)
    analytic = measure * (operator @ density)
    step = 1e-6
    for k in (5, 60, 130):
        shifted = density.copy()
        shifted[k] += step
        lowered = density.copy()
        lowered[k] -= step
        assert (energy(shifted) - energy(lowered)) / (2 * step) == pytest.approx(
            analytic[k], rel=1e-7
        )


def test_truncation_does_not_leak_into_the_quadrature():
    """A density cut off mid-mesh must not have its own step integrated.

    ``radial_poisson`` treats the charge as zero past ``upto``. The cubic panels span four
    neighbours, so running them over the zero-padded tail fits a cubic through that step
    and loses half a panel of the jump. With a real PAW density -- which is nowhere near
    zero at ``r_c`` -- that was worth several eV of one-centre energy.

    The identity used here is exact: at ``r = 0`` the ``l = 0`` potential is the whole
    outer integral, ``v_0(0) = 4 pi integral n_0 r dr``.
    """
    grid, stop = log_grid(), 329
    r = grid.r[:stop]
    density = np.exp(-0.8 * r**2) + 0.4 * np.exp(-30.0 * r)
    assert density[-1] > 0.05, "the density must not vanish at the cut, or nothing is tested"

    from_poisson = radial_poisson(grid, 0, density, upto=stop)[0] / (4.0 * np.pi)
    from_simpson = integrate(grid, np.concatenate([density * r, np.zeros(N - stop)]), upto=stop)
    assert from_poisson == pytest.approx(from_simpson, rel=1e-7)


def test_cumulative_integral_stops_where_it_is_told():
    """``upto`` must slice before integrating, never integrate and then slice."""
    grid, stop = log_grid(), 120
    values = np.cos(grid.r)
    cut = cumulative_integral(grid, values, upto=stop)
    assert cut.shape == (stop,)
    # The final panel uses a one-sided stencil, so only the interior has to agree exactly.
    assert cut[:-2] == pytest.approx(cumulative_integral(grid, values)[: stop - 2], rel=1e-12)


def test_poisson_tail_outside_the_charge_is_the_multipole_field():
    """Past the last charged point the potential is the bare multipole tail."""
    grid, stop = log_grid(), 200
    r = grid.r[:stop]
    for l in (0, 1, 2):
        density = r**l * np.exp(-2.0 * r**2)
        moment = multipole_moment(grid, l, density, upto=stop)
        potential = radial_poisson(grid, l, density, upto=stop)
        outside = grid.r[stop:]
        # The shape is the real content: outside the charge, v_l r^(l+1) is a constant.
        tail = potential[stop:] * outside ** (l + 1)
        assert tail == pytest.approx(tail[0], rel=1e-12)
        # And that constant is the multipole moment, to the accuracy of the quadratures --
        # the moment comes from Simpson and the tail from the cumulative rule.
        assert tail[0] == pytest.approx(4.0 * np.pi / (2 * l + 1) * moment, rel=1e-4)
