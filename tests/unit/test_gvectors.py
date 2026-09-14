"""Gate 4: the plane-wave count approaches the analytic asymptote."""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.basis.gvectors import GSphere, estimate_count, miller_bounds, reciprocal_cell
from amaterasu.units import ANG_IN_BOHR, parse_energy


@pytest.fixture
def si_cell_bohr(si_atoms):
    return np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR


def test_reciprocal_cell_is_biorthogonal(si_cell_bohr):
    """b_i . a_j = 2*pi delta_ij defines the 2*pi convention used throughout."""
    product = reciprocal_cell(si_cell_bohr) @ si_cell_bohr.T
    assert np.allclose(product, 2 * np.pi * np.eye(3), atol=1e-12)


def test_reciprocal_cell_rejects_singular():
    with pytest.raises(ValueError, match="singular"):
        reciprocal_cell(np.array([[1.0, 0, 0], [2.0, 0, 0], [0, 0, 1.0]]))


@pytest.mark.parametrize("ecut_ry", [60, 120, 200, 400, 800])
def test_plane_wave_count_matches_the_asymptote(si_cell_bohr, ecut_ry):
    """N -> volume * (2*ecut)^(3/2) / (6*pi^2).

    The true count oscillates about this as whole shells of G-vectors cross the cutoff,
    so the tolerance is loosened at the small cutoffs where a shell is a larger fraction
    of the total.
    """
    ecut = parse_energy(ecut_ry, "Ry")
    volume = abs(float(np.linalg.det(si_cell_bohr)))
    sphere = GSphere.build(si_cell_bohr, ecut)
    expected = estimate_count(volume, ecut)
    tolerance = 0.01 if ecut_ry >= 60 else 0.02
    assert len(sphere) == pytest.approx(expected, rel=tolerance)


def test_count_converges_as_the_cutoff_grows(si_cell_bohr):
    """The relative deviation from the asymptote must shrink with cutoff."""
    volume = abs(float(np.linalg.det(si_cell_bohr)))
    errors = []
    for ecut_ry in (100, 400, 1600):
        ecut = parse_energy(ecut_ry, "Ry")
        sphere = GSphere.build(si_cell_bohr, ecut)
        expected = estimate_count(volume, ecut)
        errors.append(abs(len(sphere) - expected) / expected)
    assert errors[-1] < errors[0]
    assert errors[-1] < 2e-3


def test_scaling_with_volume(si_cell_bohr):
    """N is proportional to the cell volume at fixed cutoff."""
    ecut = parse_energy(60, "Ry")
    small = len(GSphere.build(si_cell_bohr, ecut))
    large = len(GSphere.build(2.0 * si_cell_bohr, ecut))
    assert large / small == pytest.approx(8.0, rel=0.02)


def test_sorted_by_kinetic_energy(si_cell_bohr):
    sphere = GSphere.build(si_cell_bohr, parse_energy(60, "Ry"))
    assert np.all(np.diff(sphere.g2) >= -1e-12)
    assert np.array_equal(sphere.miller[0], [0, 0, 0])
    assert sphere.g2[0] == pytest.approx(0.0, abs=1e-20)


def test_first_shell_kinetic_energy(si_cell_bohr, si_atoms):
    """The shortest reciprocal vector of the fcc primitive cell has |b| = 2*pi*sqrt(3)/a."""
    sphere = GSphere.build(si_cell_bohr, parse_energy(60, "Ry"))
    a_bohr = 5.4307 * ANG_IN_BOHR
    expected = (2 * np.pi * np.sqrt(3) / a_bohr) ** 2 / 2
    assert sphere.kinetic_energy[1] == pytest.approx(expected, rel=1e-9)


def test_every_vector_is_inside_the_cutoff(si_cell_bohr):
    ecut = parse_energy(40, "Ry")
    sphere = GSphere.build(si_cell_bohr, ecut)
    assert np.all(sphere.kinetic_energy <= ecut + 1e-12)
    assert sphere.gmax == pytest.approx(np.sqrt(2 * ecut))


def test_miller_bounds_enclose_the_sphere(si_cell_bohr):
    """The bound must never clip the sphere: no vector may sit on the box edge unused."""
    ecut = parse_energy(60, "Ry")
    sphere = GSphere.build(si_cell_bohr, ecut)
    bounds = miller_bounds(si_cell_bohr, np.sqrt(2 * ecut))
    assert np.all(np.abs(sphere.miller).max(axis=0) <= bounds)


def test_cartesian_vectors_match_the_miller_indices(si_cell_bohr):
    sphere = GSphere.build(si_cell_bohr, parse_energy(30, "Ry"))
    reconstructed = sphere.miller @ reciprocal_cell(si_cell_bohr)
    assert np.allclose(sphere.cartesian, reconstructed, atol=1e-12)
    assert np.allclose(sphere.g2, np.einsum("ij,ij->i", sphere.cartesian, sphere.cartesian))


def test_finite_kpoint_shifts_the_sphere(si_cell_bohr):
    """At k != 0 no plane wave has zero kinetic energy, since |k+G| > 0 for every G."""
    ecut = parse_energy(60, "Ry")
    gamma = GSphere.build(si_cell_bohr, ecut)
    shifted = GSphere.build(si_cell_bohr, ecut, kpoint=(0.25, 0.25, 0.25))
    assert shifted.g2.min() > 1e-6
    assert abs(len(shifted) - len(gamma)) / len(gamma) < 0.05


def test_density_sphere_reaches_twice_the_wavefunction_g(si_cell_bohr):
    """ecut_rho = 4*ecut doubles G_max, which is what representing |psi|^2 requires."""
    ecut = parse_energy(40, "Ry")
    wfc = GSphere.build(si_cell_bohr, ecut)
    rho = GSphere.build(si_cell_bohr, 4 * ecut)
    assert np.all(np.abs(rho.miller).max(axis=0) == 2 * np.abs(wfc.miller).max(axis=0))
    assert rho.gmax == pytest.approx(2 * wfc.gmax)


def test_negative_cutoff_is_rejected(si_cell_bohr):
    with pytest.raises(ValueError, match="positive"):
        GSphere.build(si_cell_bohr, -1.0)
