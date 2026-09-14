"""Ewald summation, checked against exactly known Madelung constants.

The Madelung constant of a binary ionic crystal is a pure number fixed by the lattice
geometry, tabulated to ten digits, and completely independent of any electronic-structure
convention -- which makes it the cleanest possible reference for a lattice sum.
"""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.hamiltonian.ewald import ewald_energy, optimal_eta

#: Madelung constants alpha, defined by E = -alpha / r_nn per formula unit in Hartree
#: atomic units. Values from the standard tabulations (e.g. Kittel, Introduction to Solid
#: State Physics, Table 3.5) and reproducible by direct summation.
MADELUNG = {
    "rocksalt": 1.747564594633,
    "cesium-chloride": 1.762674773071,
    "zincblende": 1.638055053388,
}


def _binary_cell(kind: str, a: float):
    """Cell (rows, Bohr), positions (Bohr) and nearest-neighbour distance for a binary ionic crystal."""
    if kind == "rocksalt":
        cell = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
        positions = np.array([[0.0, 0, 0], [0.5 * a, 0, 0]])
        return cell, positions, 0.5 * a
    if kind == "cesium-chloride":
        cell = a * np.eye(3)
        positions = np.array([[0.0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * a]])
        return cell, positions, 0.5 * np.sqrt(3) * a
    cell = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    positions = np.array([[0.0, 0, 0], [0.25 * a, 0.25 * a, 0.25 * a]])
    return cell, positions, 0.25 * np.sqrt(3) * a


@pytest.mark.parametrize("kind", sorted(MADELUNG))
def test_madelung_constant(kind):
    cell, positions, nearest = _binary_cell(kind, 5.64)
    energy = ewald_energy(cell, positions, np.array([1.0, -1.0]))
    assert -energy * nearest == pytest.approx(MADELUNG[kind], rel=1e-9)


def test_result_is_independent_of_the_splitting_parameter():
    """eta only divides the work between the two sums; the total cannot depend on it."""
    cell, positions, _ = _binary_cell("rocksalt", 5.64)
    charges = np.array([1.0, -1.0])
    reference = ewald_energy(cell, positions, charges)
    for eta in (0.15, 0.25, 0.4, 0.7, 1.1):
        assert ewald_energy(cell, positions, charges, eta=eta) == pytest.approx(
            reference, abs=1e-11
        )


def test_translation_invariance(si_atoms):
    """Moving every ion by the same vector cannot change the electrostatic energy."""
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    positions = si_atoms.get_positions() * ANG_IN_BOHR
    charges = np.full(len(si_atoms), 4.0)
    reference = ewald_energy(cell, positions, charges)
    shifted = ewald_energy(cell, positions + np.array([0.31, -1.7, 2.4]), charges)
    assert shifted == pytest.approx(reference, abs=1e-11)


def test_periodic_image_invariance(si_atoms):
    """Adding a lattice vector to one atom must leave the energy unchanged."""
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    positions = si_atoms.get_positions() * ANG_IN_BOHR
    charges = np.full(len(si_atoms), 4.0)
    moved = positions.copy()
    moved[1] += cell[0] - 2.0 * cell[2]
    assert ewald_energy(cell, moved, charges) == pytest.approx(
        ewald_energy(cell, positions, charges), abs=1e-11
    )


def test_scaling_with_the_cell():
    """Electrostatics is homogeneous of degree -1: scaling all lengths by s scales E by 1/s."""
    cell, positions, _ = _binary_cell("rocksalt", 5.64)
    charges = np.array([1.0, -1.0])
    reference = ewald_energy(cell, positions, charges)
    scaled = ewald_energy(1.7 * cell, 1.7 * positions, charges)
    assert scaled == pytest.approx(reference / 1.7, rel=1e-10)


def test_optimal_eta_shrinks_with_the_cell():
    assert optimal_eta(1000.0) < optimal_eta(100.0)


def test_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="positions but"):
        ewald_energy(np.eye(3), np.zeros((2, 3)), np.ones(3))
