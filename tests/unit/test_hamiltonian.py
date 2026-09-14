"""The Kohn-Sham Hamiltonian: local potential, Hartree term, and the assembled operator."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from amaterasu.basis.fft import FFTGrid
from amaterasu.basis.gvectors import GSphere
from amaterasu.hamiltonian.ewald import ewald_energy
from amaterasu.hamiltonian.hartree import hartree
from amaterasu.hamiltonian.local import (
    atomic_core_density,
    atomic_valence_density,
    local_potential,
    structure_factors,
)
from amaterasu.hamiltonian.operator import Hamiltonian
from amaterasu.hamiltonian.system import build_system, default_band_count
from amaterasu.pseudo.kb import KBProjectors
from amaterasu.scf.eigen import davidson, random_bands
from amaterasu.units import ANG_IN_BOHR


# ------------------------------------------------------------------------ Hartree
def test_hartree_energy_of_gaussians_equals_the_ewald_reciprocal_sum():
    """A rigorous cross-check of hartree.py against the independent ewald.py.

    For a density of normalised Gaussians ``q_a (alpha/pi)^(3/2) exp(-alpha r^2)`` the
    Hartree sum over ``G != 0`` is *identical* to the reciprocal-space part of an Ewald
    sum of point charges taken with ``eta = sqrt(alpha/2)``, since both carry the factor
    ``exp(-G^2/2 alpha)``. Choosing alpha large enough that the Ewald real-space sum
    vanishes leaves

        E_H = E_Ewald + (eta/sqrt(pi)) sum_a q_a^2       (neutral cell)

    which ties two completely separate code paths -- an FFT on a grid and a lattice sum
    with error functions -- to the same number.
    """
    side = 12.0
    cell = side * np.eye(3)
    positions = np.array([[3.0, 3.0, 3.0], [9.0, 3.0, 3.0]])
    charges = np.array([1.0, -1.0])
    alpha = 2.0
    eta = np.sqrt(alpha / 2.0)

    grid = FFTGrid.for_cutoff(cell, 120.0)
    sphere = GSphere.build(cell, 120.0)
    points = grid.points_cartesian()
    density = np.zeros(grid.shape)
    for shift in np.stack(
        np.meshgrid(*[np.arange(-1, 2) * side] * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3):
        for charge, position in zip(charges, positions, strict=True):
            r2 = np.sum((points - position - shift) ** 2, axis=-1)
            density += charge * (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)

    energy, potential = hartree(grid, sphere, density)
    expected = ewald_energy(cell, positions, charges, eta=eta) + eta / np.sqrt(np.pi) * float(
        np.sum(charges**2)
    )
    assert energy == pytest.approx(expected, abs=1e-8)
    # And the two ways of writing the same energy must agree.
    assert 0.5 * grid.integrate(potential * density) == pytest.approx(energy, rel=1e-9)


def test_hartree_potential_solves_poisson(si_atoms, si_pseudo):
    """-G^2 V_H(G) = -4 pi n(G): the defining equation, checked component by component."""
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 40.0)
    sphere = GSphere.build(cell, 40.0)
    density = atomic_valence_density(
        grid,
        sphere,
        si_atoms.get_positions() * ANG_IN_BOHR,
        list(si_atoms.get_chemical_symbols()),
        {"Si": si_pseudo},
        total_electrons=8.0,
    )
    _, potential = hartree(grid, sphere, density)
    v_g = grid.gather(grid.to_reciprocal(potential.astype(complex)), sphere)
    n_g = grid.gather(grid.to_reciprocal(density.astype(complex)), sphere)
    nonzero = sphere.g2 > 1e-12
    residual = sphere.g2[nonzero] * v_g[nonzero] - 4.0 * np.pi * n_g[nonzero]
    assert float(np.max(np.abs(residual))) < 1e-12
    assert abs(v_g[~nonzero]).max() < 1e-14  # the G = 0 term is deliberately dropped


def test_hartree_energy_is_non_negative_and_scales_quadratically(si_atoms, si_pseudo):
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 30.0)
    sphere = GSphere.build(cell, 30.0)
    density = atomic_valence_density(
        grid,
        sphere,
        si_atoms.get_positions() * ANG_IN_BOHR,
        list(si_atoms.get_chemical_symbols()),
        {"Si": si_pseudo},
        total_electrons=8.0,
    )
    energy, _ = hartree(grid, sphere, density)
    doubled, _ = hartree(grid, sphere, 2.0 * density)
    assert energy > 0
    assert doubled == pytest.approx(4.0 * energy, rel=1e-12)


# ------------------------------------------------------------------ local potential
def test_local_potential_reproduces_the_radial_function(si_pseudo):
    """Near an isolated atom the grid potential must follow V_loc(r) from the file.

    Two corrections separate the grid result from the bare radial table, and both are
    physics rather than error:

    * dropping the ``G = 0`` divergence is exactly the statement that the ion's charge is
      neutralised by a uniform background, whose potential energy for an electron is
      ``-(2 pi Z / 3 Omega) r^2`` near the centre;
    * the alpha term then fixes the remaining additive constant, which the radial table
      cannot know about.

    Subtracting the parabola collapses the discrepancy from 21 mHa to 2 mHa over a 5 Bohr
    sphere, so the test measures the local potential *and* the ``G = 0`` convention at
    once. The 2 mHa left over is the higher (cubic-harmonic) structure of the background
    on a finite lattice, which no spherical formula reproduces.
    """
    cell = 22.0 * np.eye(3)
    grid = FFTGrid.for_cutoff(cell, 60.0)
    sphere = GSphere.build(cell, 60.0)
    centre = np.array([[11.0, 11.0, 11.0]])
    potential = local_potential(grid, sphere, centre, ["Si"], {"Si": si_pseudo})

    radius = np.linalg.norm(grid.points_cartesian() - centre[0], axis=-1)
    background = -2.0 * np.pi * si_pseudo.valence_charge / (3.0 * grid.volume) * radius**2
    reference = np.interp(radius, si_pseudo.grid.r, si_pseudo.local_potential) + background
    inside = (radius > 0.4) & (radius < 4.0)
    deviation = potential[inside] - reference[inside]
    assert float(deviation.max() - deviation.min()) < 2e-3
    # Without the background the same comparison is an order of magnitude worse.
    raw = potential[inside] - reference[inside] + background[inside]
    assert float(raw.max() - raw.min()) > 1e-2


def test_local_potential_g_zero_is_the_alpha_term(si_atoms, si_pseudo):
    """The mean of V_loc over the cell is sum_a v~_a(0) / Omega."""
    from amaterasu.pseudo.formfactors import local_potential_form_factor

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 40.0)
    sphere = GSphere.build(cell, 40.0)
    potential = local_potential(
        grid,
        sphere,
        si_atoms.get_positions() * ANG_IN_BOHR,
        list(si_atoms.get_chemical_symbols()),
        {"Si": si_pseudo},
    )
    alpha = 2.0 * local_potential_form_factor(si_pseudo, np.array([0.0]))[0]
    assert float(np.mean(potential)) == pytest.approx(alpha / grid.volume, rel=1e-10)


def test_local_potential_is_translation_covariant(si_atoms, si_pseudo):
    """Shifting every atom by a lattice vector leaves the grid potential unchanged."""
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 30.0)
    sphere = GSphere.build(cell, 30.0)
    symbols = list(si_atoms.get_chemical_symbols())
    positions = si_atoms.get_positions() * ANG_IN_BOHR
    reference = local_potential(grid, sphere, positions, symbols, {"Si": si_pseudo})
    moved = positions + cell[1] - 3.0 * cell[2]
    assert local_potential(grid, sphere, moved, symbols, {"Si": si_pseudo}) == pytest.approx(
        reference, abs=1e-10
    )


def test_structure_factor_conventions(si_atoms):
    """S_a(G) = exp(-i G.tau_a): unit modulus, and 1 at G = 0."""
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    sphere = GSphere.build(cell, 10.0)
    factors = structure_factors(sphere, si_atoms.get_positions() * ANG_IN_BOHR)
    assert factors.shape == (len(si_atoms), len(sphere))
    assert np.abs(np.abs(factors) - 1.0).max() < 1e-12
    at_gamma = np.argmin(sphere.g2)
    assert factors[:, at_gamma] == pytest.approx(np.ones(len(si_atoms)), abs=1e-12)


def test_atomic_densities_carry_the_right_charge(si_atoms, si_pseudo):
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 40.0)
    sphere = GSphere.build(cell, 40.0)
    positions = si_atoms.get_positions() * ANG_IN_BOHR
    symbols = list(si_atoms.get_chemical_symbols())
    valence = atomic_valence_density(
        grid, sphere, positions, symbols, {"Si": si_pseudo}, total_electrons=8.0
    )
    assert grid.integrate(valence) == pytest.approx(8.0, rel=1e-10)
    assert float(valence.min()) >= 0.0

    core = atomic_core_density(grid, sphere, positions, symbols, {"Si": si_pseudo})
    assert core is not None
    # The partial core is a pseudised tail, not the true core; it must not integrate to
    # the number of core electrons (10 for Si), only to the dataset's own small value.
    assert 0.5 < grid.integrate(core) < 3.0


# ----------------------------------------------------------------------- operator
def _si_system(si_atoms, si_pseudo, config, ecut=8.0):
    return build_system(si_atoms, config, {"Si": si_pseudo}, ecut, 4 * ecut)


def test_hamiltonian_is_hermitian(si_atoms, si_pseudo, si_scf_config):
    system = _si_system(si_atoms, si_pseudo, si_scf_config)
    rng = np.random.default_rng(0)
    potential = rng.normal(size=(1, *system.grid.shape))
    hamiltonian = Hamiltonian(system, potential)
    sphere = system.wave_spheres[1]
    block = rng.normal(size=(4, len(sphere))) + 1j * rng.normal(size=(4, len(sphere)))
    matrix = block.conj() @ hamiltonian.apply(block, 1).T
    assert matrix == pytest.approx(matrix.conj().T, abs=1e-11)


def test_empty_lattice_gives_free_electron_bands(si_atoms, si_pseudo, si_scf_config):
    """Gate 1: switch every ionic term off and the eigenvalues must be |k+G|^2/2 exactly.

    This is the sharpest available test of the basis machinery on its own -- the sphere,
    the sorting, the sphere-to-grid map and the eigensolver -- because the answer is known
    in closed form with no reference calculation involved.
    """
    system = _si_system(si_atoms, si_pseudo, si_scf_config, ecut=10.0)
    system.v_local[:] = 0.0
    system.projectors = [
        KBProjectors(np.zeros((0, len(s)), dtype=complex), np.zeros((0, 0)), [])
        for s in system.wave_spheres
    ]
    hamiltonian = Hamiltonian(system, np.zeros((1, *system.grid.shape)))

    for ik, sphere in enumerate(system.wave_spheres):
        result = davidson(
            lambda block, ik=ik: hamiltonian.apply(block, ik),
            random_bands(len(sphere), 10, sphere.kinetic_energy, seed=ik),
            sphere.kinetic_energy,
            tolerance=1e-12,
            max_iterations=200,
        )
        assert result.eigenvalues == pytest.approx(np.sort(sphere.kinetic_energy)[:10], abs=1e-11)


def test_operator_rejects_a_wrongly_shaped_potential(si_atoms, si_pseudo, si_scf_config):
    system = _si_system(si_atoms, si_pseudo, si_scf_config)
    with pytest.raises(ValueError, match="effective potential has shape"):
        Hamiltonian(system, np.zeros((2, *system.grid.shape)))


# ------------------------------------------------------------------------- system
def test_band_count_leaves_room_above_the_fermi_level():
    assert default_band_count(8, 1) == 8  # 4 occupied + 4
    assert default_band_count(100, 1) == 60  # 1.2 * 50
    assert default_band_count(16, 2, moment=2.2) > default_band_count(16, 2, moment=0.0)


def test_system_setup_is_self_consistent(si_atoms, si_pseudo, si_scf_config):
    system = _si_system(si_atoms, si_pseudo, si_scf_config)
    assert system.nelectrons == 8.0
    assert system.nspin == 1
    assert len(system.wave_spheres) == len(system.kpoints) == len(system.projectors)
    assert system.symmetry is not None and system.symmetry.number == 227
    # The density is built from |psi|^2, whose Miller indices are differences of two
    # indices in the same k-sphere. Every such difference must fit the grid, or a
    # high-G component folds back onto a low-G one and quietly corrupts the density.
    for sphere in system.wave_spheres:
        differences = sphere.miller[:, None, :] - sphere.miller[None, :, :]
        widest = np.abs(differences).max(axis=(0, 1))
        assert np.all(2 * widest + 1 <= np.asarray(system.grid.shape))
    assert grid_electrons(system) == pytest.approx(8.0, rel=1e-10)


def grid_electrons(system):
    return float(system.initial_density().sum()) * system.grid.cell_volume_per_point


def test_non_periodic_input_is_rejected(si_pseudo, si_scf_config):
    molecule = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=8.0 * np.eye(3), pbc=False)
    with pytest.raises(ValueError, match="3D-periodic"):
        build_system(molecule, si_scf_config, {"Si": si_pseudo}, 8.0, 32.0)


def test_initial_density_carries_the_requested_moment(si_pseudo, si_scf_config):
    """The starting spin split must integrate to the moment the input asked for.

    The natural-looking alternative -- weighting each channel by the atom's magnetic
    moment directly -- is dimensionally wrong: a moment of 2.2 muB on an atom with 16
    valence electrons is a 14% polarisation, not a factor of 2.2, and clipping the ratio
    to one starts the SCF from a fully polarised atom. That is not a slow start but a
    different calculation, so the electron count *and* the moment are both checked here.
    """
    from ase.build import bulk

    iron = bulk("Fe", "bcc", a=2.87, cubic=True)
    iron.set_initial_magnetic_moments([2.2, 2.2])
    pseudo = _load_iron()
    if pseudo is None:
        pytest.skip("the iron dataset is not available offline")

    config = si_scf_config
    config.system.spin = "polarized"
    config.pseudo.allow_xc_mismatch = True
    system = build_system(iron, config, {"Fe": pseudo}, 12.0, 48.0)
    density = system.initial_density()
    weight = system.grid.cell_volume_per_point
    assert float(density.sum()) * weight == pytest.approx(32.0, rel=1e-10)
    assert float((density[0] - density[1]).sum()) * weight == pytest.approx(4.4, rel=1e-8)
    assert float(density.min()) >= 0.0

    iron.set_initial_magnetic_moments([2.2, -2.2])
    system = build_system(iron, config, {"Fe": pseudo}, 12.0, 48.0)
    antiferro = system.initial_density()
    assert float((antiferro[0] - antiferro[1]).sum()) * weight == pytest.approx(0.0, abs=1e-9)
    # Zero *net* moment, but the two sublattices must not be identical.
    assert not np.allclose(antiferro[0], antiferro[1])


def test_initial_density_is_spin_symmetric_without_moments(si_atoms, si_pseudo, si_scf_config):
    si_scf_config.system.spin = "polarized"
    system = build_system(si_atoms, si_scf_config, {"Si": si_pseudo}, 8.0, 32.0)
    density = system.initial_density()
    assert density[0] == pytest.approx(density[1], abs=1e-15)


def _load_iron():
    """The iron dataset, if it has already been fetched; tests never download."""
    from pathlib import Path

    from amaterasu.pseudo.readers import read_pseudopotential

    path = Path("~/.amaterasu/pseudos/pseudodojo-nc-sr-pbe-standard-v0.4/Fe-sp.psp8").expanduser()
    return read_pseudopotential(path) if path.exists() else None


def test_grid_does_not_depend_on_the_k_point_symmetry_flag(si_atoms, si_pseudo, si_scf_config):
    """Two systems for the same crystal must share a grid, whatever the k-points do.

    A band-structure run uses explicit k-points and no wedge reduction, but reuses the
    potential from a symmetry-reduced SCF -- which is only possible if the two grids
    agree. The grid is therefore made commensurate with the space group regardless of
    whether the Brillouin zone is being reduced.
    """
    reduced = build_system(si_atoms, si_scf_config, {"Si": si_pseudo}, 8.0, 32.0)

    si_scf_config.kpoints.symmetry = False
    unreduced = build_system(si_atoms, si_scf_config, {"Si": si_pseudo}, 8.0, 32.0)
    assert unreduced.grid.shape == reduced.grid.shape
    assert unreduced.symmetry is None, "density symmetrisation must follow the flag"
    assert reduced.symmetry is not None
    assert len(unreduced.kpoints) > len(reduced.kpoints)
