"""The stress tensor, checked term by term against a frozen-basis finite difference.

"Frozen basis" is the exact counterpart of the analytic derivation: strain the cell, keep
the same plane-wave coefficients attached to the same Miller indices, and recompute every
energy term. Selecting the plane waves afresh by cutoff at each strain would instead
measure the analytic stress *plus* the Pulay term, which is a different quantity and is
measured separately in the physics suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from amaterasu.basis.fft import FFTGrid
from amaterasu.basis.gvectors import GSphere, reciprocal_cell
from amaterasu.config.schema import Config
from amaterasu.hamiltonian.ewald import ewald_energy
from amaterasu.hamiltonian.hartree import hartree
from amaterasu.hamiltonian.local import atomic_core_density, local_potential
from amaterasu.hamiltonian.system import build_system
from amaterasu.properties.stress import compute_stress
from amaterasu.pseudo.kb import build_projectors
from amaterasu.scf.density import compute_density
from amaterasu.scf.eigen import random_bands

ECUT = 6.0


def _strained(sphere: GSphere, cell: np.ndarray) -> GSphere:
    """The same Miller indices re-expressed in a strained cell."""
    reciprocal = reciprocal_cell(cell)
    cartesian = sphere.miller @ reciprocal + sphere.kpoint @ reciprocal
    return GSphere(
        miller=sphere.miller,
        cartesian=cartesian,
        g2=np.einsum("gi,gi->g", cartesian, cartesian),
        kpoint=sphere.kpoint,
        ecut=sphere.ecut,
        cell=cell,
    )


@pytest.fixture(scope="module")
def frozen(si_pseudo):
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
            random_bands(len(sphere), nbands, sphere.kinetic_energy, seed=23 + ik)
            for ik, sphere in enumerate(system.wave_spheres)
        ]
    ]
    occupations = np.ones((1, system.nkpoints, nbands))
    density = compute_density(system, wavefunctions, occupations)
    return system, wavefunctions, occupations, density


def _energy_terms(system, wavefunctions, occupations, density, strain):
    matrix = np.eye(3) + strain
    cell = system.cell @ matrix.T
    positions = system.positions @ matrix.T
    grid = FFTGrid(shape=system.grid.shape, cell=cell)
    sphere = _strained(system.density_sphere, cell)
    charges = np.array([system.pseudopotentials[s].valence_charge for s in system.symbols])

    # Fixed coefficients mean the density keeps its fractional shape and dilutes.
    scaled = density * (system.grid.volume / grid.volume)
    total_density = scaled.sum(axis=0)

    kinetic = 0.0
    nonlocal_energy = 0.0
    for ik, wave_sphere in enumerate(system.wave_spheres):
        strained = _strained(wave_sphere, cell)
        projectors = build_projectors(
            strained, positions, system.symbols, system.pseudopotentials, grid.volume
        )
        weights = 2.0 * system.kpoints.weights[ik] * occupations[0, ik]
        block = wavefunctions[0][ik]
        kinetic += float(np.einsum("n,ng->", weights, np.abs(block) ** 2 * (0.5 * strained.g2)))
        nonlocal_energy += projectors.energy(block, weights)

    v_local = local_potential(grid, sphere, positions, system.symbols, system.pseudopotentials)
    core = atomic_core_density(grid, sphere, positions, system.symbols, system.pseudopotentials)
    return {
        "kinetic": kinetic,
        "hartree": hartree(grid, sphere, total_density)[0],
        "local": float(np.sum(v_local * total_density)) * grid.cell_volume_per_point,
        "nonlocal": nonlocal_energy,
        "xc": system.xc.evaluate_on_grid(scaled + core[None, ...] / system.nspin, grid).energy,
        "ewald": ewald_energy(cell, positions, charges),
    }


@pytest.fixture(scope="module")
def stress_comparison(frozen):
    system, wavefunctions, occupations, density = frozen
    analytic = compute_stress(
        system, density, wavefunctions, occupations, symmetrize=False, decompose=True
    )
    # The core term is part of dE_xc/d eps; the finite difference cannot separate them.
    merged = dict(analytic)
    merged["xc"] = merged["xc"] + merged.pop("core", 0.0)

    step = 1e-5
    names = tuple(merged)
    numerical = {name: np.zeros((3, 3)) for name in names}
    volume = system.grid.volume
    for u in range(3):
        for v in range(u, 3):
            strain = np.zeros((3, 3))
            strain[u, v] += 0.5
            strain[v, u] += 0.5
            shifted = [
                _energy_terms(system, wavefunctions, occupations, density, sign * step * strain)
                for sign in (+1, -1)
            ]
            for name in names:
                value = (shifted[0][name] - shifted[1][name]) / (2 * step) / volume
                numerical[name][u, v] = numerical[name][v, u] = value
    return merged, numerical


@pytest.mark.parametrize("term", ["ewald", "kinetic", "hartree", "local", "nonlocal", "xc"])
def test_each_stress_term_matches_a_finite_difference(stress_comparison, term):
    """Gate 3, term by term, on all six independent components.

    The nonlocal term is the one worth watching: it needs the derivative of the projector
    with respect to ``k+G``, angular part included, which is why the solid harmonics carry
    analytic gradients. A missing angular term leaves the diagonal right and the shear
    components wrong, so testing only the pressure would not catch it.
    """
    analytic, numerical = stress_comparison
    assert analytic[term] == pytest.approx(numerical[term], abs=2e-9)


def test_total_stress_matches_a_finite_difference(stress_comparison):
    analytic, numerical = stress_comparison
    assert sum(analytic.values()) == pytest.approx(sum(numerical.values()), abs=5e-9)


def test_stress_is_symmetric(stress_comparison):
    analytic, _ = stress_comparison
    total = sum(analytic.values())
    assert total == pytest.approx(total.T, abs=1e-12)


def test_cubic_symmetry_makes_the_stress_isotropic(si_atoms, si_pseudo):
    """A cubic crystal can only have a hydrostatic stress; symmetrisation enforces it."""
    from amaterasu.scf.driver import run_scf

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "fixed"
    config.scf.tol_density = 1e-8
    system = build_system(si_atoms, config, {"Si": si_pseudo}, 8.0, 32.0)
    result = run_scf(system, config)
    stress = compute_stress(system, result.density, result.wavefunctions, result.occupations)
    diagonal = np.diag(stress)
    assert diagonal == pytest.approx(np.full(3, diagonal.mean()), abs=1e-14)
    assert np.abs(stress - np.diag(diagonal)).max() < 1e-14


def test_pressure_sign_convention(si_atoms, si_pseudo):
    """Compressing a solid must give a positive pressure -- one that pushes back out."""
    from amaterasu.scf.driver import run_scf
    from amaterasu.units import HARTREE_PER_BOHR3_IN_GPA

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "fixed"
    config.scf.tol_density = 1e-8

    pressures = []
    for scale in (0.94, 1.06):
        squeezed = si_atoms.copy()
        squeezed.set_cell(si_atoms.cell * scale, scale_atoms=True)
        system = build_system(squeezed, config, {"Si": si_pseudo}, 8.0, 32.0)
        result = run_scf(system, config)
        stress = compute_stress(system, result.density, result.wavefunctions, result.occupations)
        pressures.append(-np.trace(stress) / 3 * HARTREE_PER_BOHR3_IN_GPA)
    compressed, expanded = pressures
    assert compressed > expanded
    assert compressed > 0.0


def test_voigt_conversion_round_trip():
    from amaterasu.calculator import _to_voigt

    tensor = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])
    assert _to_voigt(tensor) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
