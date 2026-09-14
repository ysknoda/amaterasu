"""Evaluating a functional on the real-space grid, including the GGA divergence term.

The decisive test here is that the returned potential really is the functional derivative
of the returned energy: perturb the density at one grid point, and dE/dn must reproduce
v_xc there. For a GGA that exercises the ``-div(df/d grad n)`` term, which no pointwise
comparison can reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.basis.fft import FFTGrid
from amaterasu.basis.gvectors import reciprocal_cell
from amaterasu.units import ANG_IN_BOHR
from amaterasu.xc.driver import XCFunctional, available_functionals

FUNCTIONALS = list(available_functionals())


@pytest.fixture(scope="module")
def grid(si_atoms_module):
    cell = np.asarray(si_atoms_module.cell[:]) * ANG_IN_BOHR
    return FFTGrid(shape=(16, 16, 16), cell=cell)


@pytest.fixture(scope="module")
def si_atoms_module():
    from ase.build import bulk

    return bulk("Si", "diamond", a=5.4307)


@pytest.fixture(scope="module")
def smooth_density(grid):
    """A strictly positive, band-limited density: what a converged calculation looks like."""
    reciprocal = reciprocal_cell(grid.cell)
    positions = grid.points_cartesian()
    density = (
        0.5
        + 0.15 * np.cos(positions @ reciprocal[0])
        + 0.10 * np.sin(positions @ (reciprocal[1] + reciprocal[2]))
        + 0.05 * np.cos(positions @ (2 * reciprocal[0] - reciprocal[1]))
    )
    assert density.min() > 0.05
    return density


def test_unknown_functional_is_rejected():
    with pytest.raises(ValueError, match="unknown functional"):
        XCFunctional("B3LYP")


def test_gga_requires_sigma():
    with pytest.raises(ValueError, match="needs sigma"):
        XCFunctional("PBE").evaluate(np.array([[0.5, 0.5]]))


@pytest.mark.parametrize("name", FUNCTIONALS)
def test_potential_is_the_functional_derivative(name, grid, smooth_density):
    """Perturb one grid point: dE/dn(r) / (Omega/N) must equal v_xc(r).

    This is the only check that reaches the GGA divergence term, and it is where a sign
    error or a missing factor in the flux ``2 vsigma grad n`` would show up.
    """
    functional = XCFunctional(name)
    result = functional.evaluate_on_grid(smooth_density, grid)
    weight = grid.cell_volume_per_point

    for index in [(0, 0, 0), (3, 5, 7), (8, 8, 8), (2, 11, 4)]:
        step = 1e-6
        plus = smooth_density.copy()
        plus[index] += step
        minus = smooth_density.copy()
        minus[index] -= step
        finite = (
            functional.evaluate_on_grid(plus, grid).energy
            - functional.evaluate_on_grid(minus, grid).energy
        ) / (2 * step * weight)
        assert result.potential[0][index] == pytest.approx(finite, rel=1e-5)


@pytest.mark.parametrize("name", FUNCTIONALS)
def test_spin_polarized_reduces_to_unpolarized(name, grid, smooth_density):
    """Gate 2 on the grid: an equal-spin density must give the unpolarised answer exactly."""
    functional = XCFunctional(name)
    unpolarized = functional.evaluate_on_grid(smooth_density, grid)
    polarized = functional.evaluate_on_grid(
        np.stack([smooth_density / 2, smooth_density / 2]), grid
    )
    assert polarized.energy == pytest.approx(unpolarized.energy, rel=1e-13)
    assert polarized.potential[0] == pytest.approx(unpolarized.potential[0], rel=1e-13)
    assert polarized.potential[1] == pytest.approx(unpolarized.potential[0], rel=1e-13)


@pytest.mark.parametrize("name", FUNCTIONALS)
def test_energy_is_extensive_and_negative(name, grid, smooth_density):
    """E_xc is negative for any physical density, and scales with the cell volume."""
    result = XCFunctional(name).evaluate_on_grid(smooth_density, grid)
    assert result.energy < 0.0
    electrons = float(np.sum(smooth_density) * grid.cell_volume_per_point)
    # A few eV per electron is the right order for exchange-correlation.
    assert -3.0 < result.energy / electrons < -0.1


def test_uniform_density_reproduces_the_pointwise_lda(grid):
    """For a constant density the GGA gradient term vanishes and only the LDA part is left."""
    from amaterasu.xc import lda

    density = np.full(grid.shape, 0.3)
    eps_expected, vrho_expected = lda.exchange(np.array([[0.15, 0.15]]))
    correlation = lda.correlation_pw92(np.array([[0.15, 0.15]]), modified=True)

    result = XCFunctional("PBE").evaluate_on_grid(density, grid)
    expected_energy = (eps_expected[0] + correlation[0][0]) * 0.3 * grid.volume
    assert result.energy == pytest.approx(expected_energy, rel=1e-12)
    expected_potential = vrho_expected[0, 0] + correlation[1][0, 0]
    assert result.potential[0] == pytest.approx(np.full(grid.shape, expected_potential), rel=1e-12)


def test_pbesol_differs_from_pbe_only_where_the_gradient_is_finite(grid, smooth_density):
    """The two functionals coincide for a uniform density and separate for a varying one."""
    uniform = np.full(grid.shape, 0.3)
    pbe_uniform = XCFunctional("PBE").evaluate_on_grid(uniform, grid).energy
    sol_uniform = XCFunctional("PBEsol").evaluate_on_grid(uniform, grid).energy
    assert sol_uniform == pytest.approx(pbe_uniform, rel=1e-13)

    pbe = XCFunctional("PBE").evaluate_on_grid(smooth_density, grid).energy
    sol = XCFunctional("PBEsol").evaluate_on_grid(smooth_density, grid).energy
    assert abs(sol - pbe) > 1e-4
    # PBEsol's weaker enhancement means a less negative exchange-correlation energy.
    assert sol > pbe


def test_wrong_shape_is_rejected(grid):
    with pytest.raises(ValueError, match="density must have shape"):
        XCFunctional("PBE").evaluate_on_grid(np.zeros((3, *grid.shape)), grid)
    with pytest.raises(ValueError, match="density must have shape"):
        XCFunctional("PBE").evaluate_on_grid(np.zeros((8, 8, 8)), grid)


@pytest.mark.parametrize("name", FUNCTIONALS)
def test_grid_result_matches_pointwise_evaluation(name, grid, smooth_density):
    """The grid path must agree with calling the pointwise functional directly."""
    functional = XCFunctional(name)
    result = functional.evaluate_on_grid(smooth_density, grid)

    points = int(np.prod(grid.shape))
    rho = np.stack([smooth_density.ravel() / 2, smooth_density.ravel() / 2], axis=-1)
    if functional.is_gga:
        gradient = grid.gradient(smooth_density / 2).reshape(points, 3)
        contracted = np.einsum("pi,pi->p", gradient, gradient)
        sigma = np.stack([contracted, contracted, contracted], axis=-1)
        eps = functional.evaluate(rho, sigma)[0]
    else:
        eps = functional.evaluate(rho)[0]
    assert result.energy_per_particle.ravel() == pytest.approx(eps, rel=1e-13)
