"""Forces, checked against finite differences of the energy they are the derivative of.

The Hellmann-Feynman force is ``dE/d tau`` at *fixed* orbital coefficients. That identity
holds for any consistent pair of orbitals and density, not only for a converged one, so
these tests build a random orthonormal set of bands, form the density they imply, and
difference the energy terms directly -- no self-consistent field cycle in the loop, and
therefore no convergence tolerance to hide behind.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from amaterasu.config.schema import Config
from amaterasu.hamiltonian.ewald import ewald, ewald_energy
from amaterasu.hamiltonian.local import atomic_core_density, local_potential
from amaterasu.hamiltonian.system import build_system
from amaterasu.properties.forces import compute_forces
from amaterasu.pseudo.kb import build_projectors
from amaterasu.scf.density import compute_density
from amaterasu.scf.eigen import random_bands
from amaterasu.units import ANG_IN_BOHR

ECUT = 6.0


@pytest.fixture(scope="module")
def frozen(si_pseudo):
    """A low-symmetry cell with a random but self-consistent orbital/density pair."""
    cell = np.array([[5.1, 0.0, 0.0], [0.6, 4.8, 0.0], [-0.4, 0.5, 5.4]])
    positions = np.array([[0.11, 0.07, -0.05], [1.95, 1.15, 1.55]])
    atoms = Atoms("Si2", positions=positions, cell=cell, pbc=True)

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    system = build_system(atoms, config, {"Si": si_pseudo}, ECUT, 4 * ECUT)

    nbands = 4
    wavefunctions = [
        [
            random_bands(len(sphere), nbands, sphere.kinetic_energy, seed=17 + ik)
            for ik, sphere in enumerate(system.wave_spheres)
        ]
    ]
    occupations = np.zeros((1, system.nkpoints, nbands))
    occupations[0, :, :] = 1.0
    density = compute_density(system, wavefunctions, occupations)
    return system, wavefunctions, occupations, density


def _energy_terms(system, positions, density, wavefunctions, occupations):
    """The position-dependent energy terms at fixed orbitals, in Hartree."""
    grid, sphere = system.grid, system.density_sphere
    charges = np.array([system.pseudopotentials[s].valence_charge for s in system.symbols])
    total_density = density.sum(axis=0)

    v_local = local_potential(grid, sphere, positions, system.symbols, system.pseudopotentials)
    core = atomic_core_density(grid, sphere, positions, system.symbols, system.pseudopotentials)
    xc_density = density + core[None, ...] / system.nspin

    nonlocal_energy = 0.0
    for ik, wave_sphere in enumerate(system.wave_spheres):
        projectors = build_projectors(
            wave_sphere, positions, system.symbols, system.pseudopotentials, grid.volume
        )
        weights = 2.0 * system.kpoints.weights[ik] * occupations[0, ik]
        nonlocal_energy += projectors.energy(wavefunctions[0][ik], weights)

    return {
        "local": float(np.sum(v_local * total_density)) * grid.cell_volume_per_point,
        "core": system.xc.evaluate_on_grid(xc_density, grid).energy,
        "nonlocal": nonlocal_energy,
        "ewald": ewald_energy(system.cell, positions, charges),
    }


@pytest.fixture(scope="module")
def force_comparison(frozen):
    """Analytic force contributions and their finite-difference counterparts."""
    system, wavefunctions, occupations, density = frozen
    analytic = compute_forces(
        system, density, wavefunctions, occupations, symmetrize=False, decompose=True
    )
    names = ("ewald", "local", "core", "nonlocal")
    step = 3e-4
    numerical = {name: np.zeros((len(system.symbols), 3)) for name in names}
    for atom in range(len(system.symbols)):
        for axis in range(3):
            shifted = []
            for sign in (+1, -1):
                positions = system.positions.copy()
                positions[atom, axis] += sign * step
                shifted.append(
                    _energy_terms(system, positions, density, wavefunctions, occupations)
                )
            for name in names:
                numerical[name][atom, axis] = -(shifted[0][name] - shifted[1][name]) / (2 * step)
    return analytic, numerical


@pytest.mark.parametrize("term", ["ewald", "local", "core", "nonlocal"])
def test_each_force_term_matches_a_finite_difference(force_comparison, term):
    """Gate 1, term by term: every contribution is the derivative it claims to be.

    Splitting the check by term is what makes a failure diagnosable. A wrong sign in the
    nonlocal projector phase and a wrong prefactor in the core correction both show up as
    "the force is a bit off" in the total, and as an exact statement about one term here.
    """
    analytic, numerical = force_comparison
    assert analytic[term] == pytest.approx(numerical[term], abs=2e-7)


def test_total_force_matches_a_finite_difference(force_comparison):
    analytic, numerical = force_comparison
    total = analytic.total
    reference = sum(numerical.values())
    # 1e-3 eV/Angstrom is the gate; this is four orders below it.
    assert total == pytest.approx(reference, abs=1e-7)


def test_forces_sum_to_zero_for_a_converged_calculation(si_atoms, si_pseudo):
    """Gate 2: translational invariance. Necessary, not sufficient -- hence the test above."""
    from amaterasu.scf.driver import run_scf

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "fixed"
    config.scf.tol_energy = 1e-10
    config.scf.tol_density = 1e-9
    displaced = si_atoms.copy()
    displaced.positions[1] += np.array([0.06, -0.04, 0.03])
    system = build_system(displaced, config, {"Si": si_pseudo}, 8.0, 32.0)
    result = run_scf(system, config)
    assert result.converged

    forces = compute_forces(system, result.density, result.wavefunctions, result.occupations)
    from amaterasu.units import HARTREE_PER_BOHR_IN_EV_PER_ANG

    residual = np.abs(forces.sum(axis=0)).max() * HARTREE_PER_BOHR_IN_EV_PER_ANG
    assert residual < 1e-6
    # Displacing one atom of a diamond cell along a general direction must produce a
    # restoring force; a zero here would mean the calculation is not seeing the move.
    assert np.abs(forces).max() * HARTREE_PER_BOHR_IN_EV_PER_ANG > 1e-3


def test_forces_vanish_on_a_high_symmetry_site(si_atoms, si_pseudo):
    """Symmetrisation makes an ideal diamond lattice give exactly zero force.

    Without it, the irreducible-wedge k-point sum leaves a residue that has the symmetry
    of the wedge rather than of the crystal.
    """
    from amaterasu.scf.driver import run_scf

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "fixed"
    config.scf.tol_density = 1e-8
    system = build_system(si_atoms, config, {"Si": si_pseudo}, 8.0, 32.0)
    result = run_scf(system, config)
    forces = compute_forces(system, result.density, result.wavefunctions, result.occupations)
    assert np.abs(forces).max() < 1e-14


def test_ewald_forces_and_stress_against_finite_differences():
    """The ion-ion sum has both derivatives in closed form; check them together."""
    rng = np.random.default_rng(0)
    cell = np.array([[7.0, 0.3, -0.2], [0.4, 6.2, 0.5], [-0.1, 0.6, 8.1]])
    positions = rng.uniform(0, 1, (4, 3)) @ cell
    charges = np.array([3.0, -1.0, -1.0, -1.0])
    result = ewald(cell, positions, charges)
    assert np.abs(result.forces.sum(axis=0)).max() < 1e-12

    step = 1e-5
    numerical = np.zeros_like(positions)
    for atom in range(len(positions)):
        for axis in range(3):
            moved = [positions.copy(), positions.copy()]
            moved[0][atom, axis] += step
            moved[1][atom, axis] -= step
            numerical[atom, axis] = -(
                ewald_energy(cell, moved[0], charges) - ewald_energy(cell, moved[1], charges)
            ) / (2 * step)
    assert result.forces == pytest.approx(numerical, abs=1e-8)

    volume = abs(float(np.linalg.det(cell)))
    numerical_stress = np.zeros((3, 3))
    for u in range(3):
        for v in range(3):
            strain = np.zeros((3, 3))
            strain[u, v] += 0.5
            strain[v, u] += 0.5
            energies = []
            for sign in (+1, -1):
                matrix = np.eye(3) + sign * step * strain
                energies.append(ewald_energy(cell @ matrix.T, positions @ matrix.T, charges))
            numerical_stress[u, v] = (energies[0] - energies[1]) / (2 * step) / volume
    assert result.stress == pytest.approx(numerical_stress, abs=1e-10)


def test_force_symmetrisation_is_consistent_with_the_structure(si_atoms, si_pseudo):
    from amaterasu.system.symmetry import analyze, atom_mapping

    config = Config(structure="-", solver="scf")
    config.kpoints.grid = [2, 2, 2]
    system = build_system(si_atoms, config, {"Si": si_pseudo}, ECUT, 4 * ECUT)
    info = analyze(si_atoms)
    mapping = atom_mapping(si_atoms, info)
    assert mapping.shape == (info.n_operations, len(si_atoms))
    # Every operation is a bijection on the atoms.
    for row in mapping:
        assert sorted(row.tolist()) == list(range(len(si_atoms)))
    del system


def test_atom_mapping_rejects_an_inconsistent_structure(si_atoms):
    from amaterasu.system.symmetry import analyze, atom_mapping

    info = analyze(si_atoms)
    broken = si_atoms.copy()
    broken.positions[1] += np.array([0.3, 0.0, 0.0])
    with pytest.raises(ValueError, match="does not map the structure onto itself"):
        atom_mapping(broken, info)


def test_cell_conversion_is_applied_once(si_atoms, si_pseudo):
    """The system stores Bohr; a stray Angstrom would show up as a factor of 1.89."""
    config = Config(structure="-", solver="scf")
    config.kpoints.grid = [1, 1, 1]
    system = build_system(si_atoms, config, {"Si": si_pseudo}, ECUT, 4 * ECUT)
    assert system.cell == pytest.approx(np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR)
