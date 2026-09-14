"""Gate 6: the FFT pair is exactly inverse, correctly normalised, and alias-safe."""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.basis.fft import FFTGrid, available_backends, fast_grid_shape
from amaterasu.basis.gvectors import GSphere
from amaterasu.units import ANG_IN_BOHR, parse_energy


@pytest.fixture
def si_cell_bohr(si_atoms):
    return np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR


@pytest.fixture
def spheres_and_grid(si_cell_bohr):
    ecut = parse_energy(40, "Ry")
    wfc = GSphere.build(si_cell_bohr, ecut)
    rho = GSphere.build(si_cell_bohr, 4 * ecut)
    return wfc, rho, FFTGrid.for_sphere(rho)


def test_fast_grid_shape_rounds_up():
    assert fast_grid_shape((17, 23, 31)) == (18, 24, 32)
    for n in fast_grid_shape((13, 29, 61)):
        remaining = n
        for radix in (2, 3, 5, 7, 11):
            while remaining % radix == 0:
                remaining //= radix
        assert remaining == 1, f"{n} is not a fast FFT length"


def test_grid_is_large_enough_for_its_sphere(spheres_and_grid):
    _, rho, grid = spheres_and_grid
    h_max = np.abs(rho.miller).max(axis=0)
    assert np.all(np.asarray(grid.shape) >= 2 * h_max + 1)


def test_scatter_gather_round_trip(spheres_and_grid):
    wfc, _, grid = spheres_and_grid
    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=len(wfc)) + 1j * rng.normal(size=len(wfc))
    assert np.array_equal(grid.gather(grid.scatter(coefficients, wfc), wfc), coefficients)


def test_fft_round_trip(spheres_and_grid):
    """Gate 6: the transform pair must be inverse to better than 1e-13."""
    wfc, _, grid = spheres_and_grid
    rng = np.random.default_rng(1)
    coefficients = rng.normal(size=len(wfc)) + 1j * rng.normal(size=len(wfc))
    recovered = grid.real_to_wave(grid.wave_to_real(coefficients, wfc), wfc)
    assert np.abs(recovered - coefficients).max() < 1e-13


def test_normalisation_against_an_analytic_function(spheres_and_grid):
    """u(r) = cos(G.r) has exactly c_(+G) = c_(-G) = 1/2 and nothing else."""
    wfc, _, grid = spheres_and_grid
    index = 5
    g_vector = wfc.cartesian[index]
    values = np.cos(grid.points_cartesian() @ g_vector)
    coefficients = grid.to_reciprocal(values)

    shape = np.asarray(grid.shape)
    plus = tuple(wfc.miller[index] % shape)
    minus = tuple((-wfc.miller[index]) % shape)
    assert coefficients[plus].real == pytest.approx(0.5, abs=1e-12)
    assert coefficients[minus].real == pytest.approx(0.5, abs=1e-12)

    residual = np.abs(coefficients).sum() - abs(coefficients[plus]) - abs(coefficients[minus])
    assert residual < 1e-11


def test_parseval(spheres_and_grid):
    """(Omega/N) sum_r |u|^2 == Omega sum_G |c_G|^2 fixes the normalisation completely."""
    wfc, _, grid = spheres_and_grid
    rng = np.random.default_rng(2)
    coefficients = rng.normal(size=len(wfc)) + 1j * rng.normal(size=len(wfc))
    values = grid.wave_to_real(coefficients, wfc)
    real_space = float(np.real(grid.integrate(np.abs(values) ** 2)))
    reciprocal = grid.volume * float(np.sum(np.abs(coefficients) ** 2))
    assert real_space == pytest.approx(reciprocal, rel=1e-12)


def test_integrating_a_constant_gives_the_volume(spheres_and_grid):
    _, _, grid = spheres_and_grid
    assert grid.integrate(np.ones(grid.shape)).real == pytest.approx(grid.volume, rel=1e-12)
    assert grid.cell_volume_per_point * grid.size == pytest.approx(grid.volume, rel=1e-12)


def test_too_small_a_grid_raises_rather_than_aliasing(si_cell_bohr, spheres_and_grid):
    """Aliasing is silent by nature, so it must be an error rather than a quiet wrong answer."""
    _, rho, _ = spheres_and_grid
    small = FFTGrid(shape=(8, 8, 8), cell=si_cell_bohr)
    with pytest.raises(ValueError, match="alias"):
        small.scatter(np.ones(len(rho)), rho)


def test_batched_transforms(spheres_and_grid):
    wfc, _, grid = spheres_and_grid
    rng = np.random.default_rng(3)
    batch = rng.normal(size=(4, len(wfc))) + 1j * rng.normal(size=(4, len(wfc)))
    values = grid.wave_to_real(batch, wfc)
    assert values.shape == (4, *grid.shape)
    assert np.abs(grid.real_to_wave(values, wfc) - batch).max() < 1e-13
    # Each band must transform independently of the others.
    single = grid.wave_to_real(batch[2], wfc)
    assert np.abs(values[2] - single).max() < 1e-14


def test_grid_points_are_uniform_in_fractional_coordinates(spheres_and_grid):
    _, _, grid = spheres_and_grid
    fractional = grid.points_fractional()
    assert fractional.shape == (*grid.shape, 3)
    assert np.all(fractional >= 0.0)
    assert np.all(fractional < 1.0)
    assert np.allclose(
        fractional.reshape(-1, 3) @ grid.cell, grid.points_cartesian().reshape(-1, 3)
    )


def test_for_cutoff_matches_for_sphere(si_cell_bohr):
    ecut_rho = parse_energy(160, "Ry")
    sphere = GSphere.build(si_cell_bohr, ecut_rho)
    assert FFTGrid.for_cutoff(si_cell_bohr, ecut_rho).shape == FFTGrid.for_sphere(sphere).shape


@pytest.mark.parametrize("backend", available_backends())
def test_backends_agree(si_cell_bohr, backend):
    """pyFFTW must give the same answer as scipy, or it is not an optimisation."""
    ecut = parse_energy(40, "Ry")
    wfc = GSphere.build(si_cell_bohr, ecut)
    rng = np.random.default_rng(4)
    coefficients = rng.normal(size=len(wfc)) + 1j * rng.normal(size=len(wfc))

    reference = FFTGrid.for_cutoff(si_cell_bohr, 4 * ecut, backend="scipy")
    candidate = FFTGrid.for_cutoff(si_cell_bohr, 4 * ecut, backend=backend)
    assert (
        np.abs(
            candidate.wave_to_real(coefficients, wfc) - reference.wave_to_real(coefficients, wfc)
        ).max()
        < 1e-12
    )
