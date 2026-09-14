"""Phase 3 gates 1-3 and 6: forces and stress against full self-consistent finite differences.

The unit suite already checks every analytic derivative against a *frozen-orbital* finite
difference, which is the exact statement of what was derived. What that cannot see is
whether holding the orbitals fixed was legitimate in the first place, and whether the
plane-wave basis is complete enough for the result to be the physical derivative. Those
need a converged calculation at each displaced or strained geometry, which is what this
file does -- and why it is slow enough to live outside the default suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from amaterasu.calculator import Amaterasu
from amaterasu.config.schema import Config
from amaterasu.units import BOHR_IN_ANG, HARTREE_IN_EV, HARTREE_PER_BOHR3_IN_GPA

from ..conftest import PSEUDO_DATA

pytestmark = [pytest.mark.physics, pytest.mark.slow]

CELL = np.array([[4.6, 0.0, 0.0], [0.5, 4.3, 0.0], [-0.3, 0.4, 4.9]])
POSITIONS = np.array([[0.10, 0.05, -0.07], [1.75, 1.05, 1.40], [2.6, 2.9, 3.1]])
STRESS_TO_GPA = HARTREE_PER_BOHR3_IN_GPA / (HARTREE_IN_EV / BOHR_IN_ANG**3)


def _make(displacement=None, strain=None, ecut=40.0):
    matrix = np.eye(3) if strain is None else np.eye(3) + strain
    positions = POSITIONS if displacement is None else POSITIONS + displacement
    atoms = Atoms("Si3", positions=positions @ matrix.T, cell=CELL @ matrix.T, pbc=True)

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "gaussian"
    config.occupations.smearing = 0.2
    config.pseudo.directory = str(PSEUDO_DATA)
    config.pseudo.files = {"Si": "Si.pbe.psp8.gz"}
    config.basis.ecut = ecut  # Rydberg
    config.scf.mixing.preconditioner = "none"
    config.scf.max_steps = 100
    config.scf.tol_energy = 1e-11
    config.scf.tol_density = 1e-10
    atoms.calc = Amaterasu(config, download=False)
    return atoms


def _free_energy(atoms) -> float:
    """The force-consistent energy, in eV.

    This is the one the forces are the gradient of. Differencing the ``sigma -> 0``
    extrapolation instead -- which is what ``get_potential_energy()`` returns -- is off by
    half the derivative of the entropy term, and on this cell that is a 6 meV/Angstrom
    error: small enough to look like a convergence problem and large enough to fail the
    gate.
    """
    atoms.get_potential_energy()
    return atoms.calc.scf.energies.free_energy * HARTREE_IN_EV


@pytest.fixture(scope="module")
def reference():
    atoms = _make()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    return atoms, forces, stress


def test_forces_match_a_full_scf_finite_difference(reference):
    """Gate 1: ``F = -dE/dR`` to better than 1e-3 eV/Angstrom on a low-symmetry cell."""
    _, forces, _ = reference
    step = 0.002
    numerical = np.zeros_like(forces)
    for atom in range(len(POSITIONS)):
        for axis in range(3):
            displacement = np.zeros_like(POSITIONS)
            displacement[atom, axis] = step
            energies = [_free_energy(_make(sign * displacement)) for sign in (+1, -1)]
            numerical[atom, axis] = -(energies[0] - energies[1]) / (2 * step)
    assert forces == pytest.approx(numerical, abs=1e-3)
    # In practice it is far better than the gate; assert that too, so a regression that
    # merely stays inside the tolerance still shows up.
    assert np.abs(forces - numerical).max() < 5e-5


def test_forces_sum_to_zero(reference):
    """Gate 2: translational invariance, to 1e-6 eV/Angstrom."""
    _, forces, _ = reference
    assert np.abs(forces.sum(axis=0)).max() < 1e-6


def test_stress_matches_a_full_scf_finite_difference_and_shows_the_pulay_term(reference):
    """Gates 3 and 6: all six components, with the Pulay stress separated out.

    The frozen-basis check in the unit suite agrees to 1e-8 GPa, so whatever is left here
    is the Pulay stress: the plane-wave set is defined by ``|k+G|^2/2 <= ecut`` in the
    strained cell, so straining it changes which functions exist -- a discontinuity the
    analytic derivative cannot see. It shrinks with the cutoff, which is the only way to
    tell it apart from an error in the derivation.
    """
    atoms, _, stress = reference
    volume = atoms.get_volume() / BOHR_IN_ANG**3
    step = 1e-4

    def measure(ecut):
        strained_stress = _make(ecut=ecut).get_stress() * STRESS_TO_GPA
        numerical = np.zeros((3, 3))
        for u in range(3):
            for v in range(u, 3):
                strain = np.zeros((3, 3))
                strain[u, v] += 0.5
                strain[v, u] += 0.5
                energies = [_free_energy(_make(strain=sign * step * strain)) for sign in (+1, -1)]
                value = (energies[0] - energies[1]) / (2 * step) / volume
                numerical[u, v] = numerical[v, u] = value
        numerical *= HARTREE_PER_BOHR3_IN_GPA / HARTREE_IN_EV
        analytic = np.array(
            [
                [strained_stress[0], strained_stress[5], strained_stress[4]],
                [strained_stress[5], strained_stress[1], strained_stress[3]],
                [strained_stress[4], strained_stress[3], strained_stress[2]],
            ]
        )
        return analytic, numerical

    coarse_analytic, coarse_numerical = measure(24.0)
    fine_analytic, fine_numerical = measure(40.0)
    del stress

    coarse_pulay = -np.trace(coarse_analytic - coarse_numerical) / 3
    fine_pulay = -np.trace(fine_analytic - fine_numerical) / 3
    # The discrepancy is a basis-set effect, so it must shrink when the basis grows. On
    # this cell it goes -0.080 GPa (24 Ry) -> -0.0027 (40 Ry) -> -0.0007 (60 Ry); the two
    # cheapest of those are enough to show the trend without a nightly job that runs for
    # hours, since the 48 and 60 Ry points cost several times as much for no new
    # information.
    assert abs(fine_pulay) < 0.2 * abs(coarse_pulay)
    assert abs(fine_pulay) < 0.02  # GPa
    assert np.abs(fine_analytic - fine_numerical).max() < 0.02
