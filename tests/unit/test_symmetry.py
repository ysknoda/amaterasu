"""Gate 5: space groups and irreducible k-point counts match known values."""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.config.schema import KpointsConfig
from amaterasu.system.kpoints import band_path, generate, grid_from_density
from amaterasu.system.symmetry import analyze, ibz_kpoints, symprec_scan


def test_silicon_space_group(si_atoms):
    info = analyze(si_atoms)
    assert info.number == 227
    assert info.international == "Fd-3m"
    assert info.pointgroup == "m-3m"
    assert info.n_operations == 48
    assert info.n_inequivalent_atoms == 1


def test_rocksalt_space_group(nacl_atoms):
    info = analyze(nacl_atoms)
    assert info.number == 225
    assert info.international == "Fm-3m"
    assert info.n_operations == 48


def test_rutile_space_group(rutile_atoms):
    """Rutile TiO2, 6 atoms per cell; the shortest Ti-O bond is 1.9486 A (exp 1.949)."""
    info = analyze(rutile_atoms)
    assert info.number == 136
    assert info.international == "P4_2/mnm"
    assert info.n_operations == 16
    assert len(rutile_atoms) == 6

    distances = rutile_atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() == pytest.approx(1.9486, abs=1e-3)


def test_symprec_scan_reports_a_stable_structure(si_atoms):
    """An ideal structure keeps its space group across every reasonable tolerance."""
    assert {number for _, number, _ in symprec_scan(si_atoms)} == {227}


def test_symprec_scan_detects_a_near_boundary_structure(si_atoms):
    """A slightly displaced atom is P1 at tight tolerance and Fd-3m at loose tolerance."""
    perturbed = si_atoms.copy()
    positions = perturbed.get_positions()
    positions[1, 0] += 1e-3
    perturbed.set_positions(positions)
    numbers = {number for _, number, _ in symprec_scan(perturbed)}
    assert len(numbers) > 1, "the scan must reveal that symprec matters here"
    assert analyze(perturbed, symprec=1e-6).number < analyze(perturbed, symprec=1e-2).number


@pytest.mark.parametrize(
    ("mesh", "shift", "expected"),
    [
        # Textbook irreducible counts for the fcc Brillouin zone. The shifted 4x4x4 set is
        # the classic Chadi-Cohen 10 special points.
        ((4, 4, 4), (0.0, 0.0, 0.0), 8),
        ((4, 4, 4), (0.5, 0.5, 0.5), 10),
        ((6, 6, 6), (0.0, 0.0, 0.0), 16),
        ((8, 8, 8), (0.0, 0.0, 0.0), 29),
        ((12, 12, 12), (0.0, 0.0, 0.0), 72),
    ],
)
def test_fcc_irreducible_kpoint_counts(si_atoms, mesh, shift, expected):
    points, weights = ibz_kpoints(si_atoms, mesh, shift=shift)
    assert len(points) == expected
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_weights_sum_to_one_and_match_the_full_mesh(si_atoms):
    """Symmetry reduction must conserve the total integration weight."""
    for mesh in [(3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)]:
        reduced, weights = ibz_kpoints(si_atoms, mesh)
        full, full_weights = ibz_kpoints(si_atoms, mesh, use_symmetry=False)
        assert len(full) == np.prod(mesh)
        assert weights.sum() == pytest.approx(full_weights.sum(), abs=1e-12)
        assert len(reduced) <= len(full)


def test_lower_symmetry_gives_more_kpoints(si_atoms, rutile_atoms):
    """Rutile (16 operations) cannot reduce a mesh as far as silicon (48) does."""
    n_si = len(ibz_kpoints(si_atoms, (8, 8, 8))[0])
    n_rutile = len(ibz_kpoints(rutile_atoms, (8, 8, 8))[0])
    assert n_rutile > n_si


def test_kpoints_lie_in_the_first_zone(si_atoms):
    points, _ = ibz_kpoints(si_atoms, (7, 7, 7))
    assert np.all(points >= -0.5 - 1e-12)
    assert np.all(points <= 0.5 + 1e-12)


def test_invalid_mesh_is_rejected(si_atoms):
    with pytest.raises(ValueError, match="positive integers"):
        ibz_kpoints(si_atoms, (0, 4, 4))
    with pytest.raises(ValueError, match="three positive integers"):
        ibz_kpoints(si_atoms, (4, 4))


def test_grid_from_density(si_atoms):
    """|b_i| = 2*pi*sqrt(3)/a = 2.0042 1/A for the fcc primitive cell of Si."""
    b_length = 2 * np.pi * np.sqrt(3) / 5.4307
    for spacing in (0.5, 0.3, 0.2, 0.1):
        expected = int(np.ceil(b_length / spacing - 1e-9))
        assert grid_from_density(si_atoms, spacing) == (expected,) * 3
    with pytest.raises(ValueError, match="positive"):
        grid_from_density(si_atoms, 0.0)


def test_generate_dispatches_on_scheme(si_atoms):
    gamma = generate(si_atoms, KpointsConfig(scheme="gamma"))
    assert len(gamma) == 1
    assert np.allclose(gamma.points, 0.0)

    mp = generate(si_atoms, KpointsConfig(grid=[8, 8, 8]))
    assert len(mp) == 29
    assert mp.n_full == 512
    assert mp.symmetry_reduced

    unreduced = generate(si_atoms, KpointsConfig(grid=[4, 4, 4], symmetry=False))
    assert len(unreduced) == 64

    density = generate(si_atoms, KpointsConfig(scheme="density", density=0.3))
    assert density.mesh == (7, 7, 7)

    explicit = generate(si_atoms, KpointsConfig(scheme="explicit", points=[[0, 0, 0], [0.5, 0, 0]]))
    assert len(explicit) == 2
    assert explicit.weights.sum() == pytest.approx(1.0)


def test_seekpath_band_path_for_fcc(si_atoms):
    """seekpath returns the standard fcc path (Hinuma et al. 2017)."""
    path = band_path(si_atoms, npoints=50)
    assert "G" in path.path
    for label in ("X", "L", "W", "K"):
        assert label in path.special_points
    assert len(path.kpts) == 50


def test_symmetrize_sphere_is_idempotent_and_charge_preserving(si_atoms):
    """Averaging over the group twice is the same as once, and G = 0 is untouched.

    ``G = 0`` maps to itself under every rotation with a trivial phase, so the electron
    count survives symmetrisation exactly -- which is what lets the SCF symmetrise a
    density without renormalising it.
    """
    import numpy as np

    from amaterasu.basis.fft import FFTGrid
    from amaterasu.basis.gvectors import GSphere
    from amaterasu.system.symmetry import analyze, symmetrize_sphere
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 20.0)
    sphere = GSphere.build(cell, 20.0)
    info = analyze(si_atoms)

    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=len(sphere)) + 1j * rng.normal(size=len(sphere))
    once = symmetrize_sphere(coefficients, sphere, grid, info.rotations, info.translations)
    twice = symmetrize_sphere(once, sphere, grid, info.rotations, info.translations)
    assert twice == pytest.approx(once, abs=1e-12)

    at_gamma = int(np.argmin(sphere.g2))
    assert once[at_gamma] == pytest.approx(coefficients[at_gamma], abs=1e-12)


def test_symmetrize_sphere_makes_a_field_invariant(si_atoms):
    """A symmetrised field is unchanged by each operation individually."""
    import numpy as np

    from amaterasu.basis.fft import FFTGrid
    from amaterasu.basis.gvectors import GSphere
    from amaterasu.system.symmetry import analyze, symmetrize_sphere
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 12.0)
    sphere = GSphere.build(cell, 12.0)
    info = analyze(si_atoms)
    rng = np.random.default_rng(1)
    field = symmetrize_sphere(
        rng.normal(size=len(sphere)) + 1j * rng.normal(size=len(sphere)),
        sphere,
        grid,
        info.rotations,
        info.translations,
    )
    for rotation, translation in zip(info.rotations[:8], info.translations[:8], strict=True):
        single = symmetrize_sphere(field, sphere, grid, rotation[None], translation[None])
        assert single == pytest.approx(field, abs=1e-12)


def test_spin_decoration_splits_an_antiferromagnet(si_atoms):
    """Two iron sublattices with opposite moments are not symmetry-equivalent.

    Without the decoration spglib returns Im-3m (229, 96 operations) and the wedge sum
    would average the two moments away; with it the magnetic cell is Pm-3m (221, 48).
    """
    from ase.build import bulk

    from amaterasu.system.symmetry import analyze, spin_decorated

    iron = bulk("Fe", "bcc", a=2.87, cubic=True)
    iron.set_initial_magnetic_moments([2.2, -2.2])
    assert analyze(iron).number == 229
    assert analyze(spin_decorated(iron)).number == 221
    # A ferromagnet has nothing to split, so the object is passed straight through.
    iron.set_initial_magnetic_moments([2.2, 2.2])
    assert spin_decorated(iron) is iron
