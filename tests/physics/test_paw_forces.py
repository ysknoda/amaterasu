"""PAW forces and stress against full self-consistent finite differences.

The norm-conserving derivatives are checked twice: once against a frozen-orbital finite
difference in the unit suite, which is the exact statement of what was derived, and once
against a fully converged finite difference here. PAW gets only the second of those, and
for a reason: the overlap operator moves with the atoms, so the orbitals cannot be held
fixed without also violating the constraint they satisfy. The term that accounts for
exactly that -- ``-eps_n O^a_ij`` in Eq. (50) of Mortensen et al. -- is the one a frozen
finite difference cannot see, and it is the same size as the force itself. Only a
self-consistent difference tests it.

Needs the JTH PAW table, which is GPL-3.0 and so not bundled; skips without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from amaterasu.calculator import Amaterasu
from amaterasu.config.schema import Config
from amaterasu.units import BOHR_IN_ANG, HARTREE_IN_EV, HARTREE_PER_BOHR3_IN_GPA

pytestmark = [pytest.mark.physics, pytest.mark.slow]

#: A deliberately low-symmetry cell: every force component is non-zero, so nothing can
#: pass by accident. Three silicon atoms in a triclinic box.
CELL = np.array([[4.6, 0.0, 0.0], [0.5, 4.3, 0.0], [-0.3, 0.4, 4.9]])
POSITIONS = np.array([[0.10, 0.05, -0.07], [1.75, 1.05, 1.40], [2.6, 2.9, 3.1]])
STRESS_TO_GPA = HARTREE_PER_BOHR3_IN_GPA / (HARTREE_IN_EV / BOHR_IN_ANG**3)

DATASET = Path("~/.amaterasu/pseudos/jth-pbe-v2.0/Si/Si.GGA_PBE-JTH.xml").expanduser()


def _make(displacement=None, strain=None, ecut=40.0):
    if not DATASET.exists():
        pytest.skip(
            f"the JTH PAW dataset is not installed at {DATASET}; fetch it with "
            "'amaterasu fetch-pseudos --family jth-pbe-v2.0 --elements Si'"
        )
    matrix = np.eye(3) if strain is None else np.eye(3) + strain
    positions = POSITIONS if displacement is None else POSITIONS + displacement
    atoms = Atoms("Si3", positions=positions @ matrix.T, cell=CELL @ matrix.T, pbc=True)

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "gaussian"
    config.occupations.smearing = 0.2
    config.pseudo.method = "paw"
    config.pseudo.files = {"Si": str(DATASET)}
    config.basis.ecut = ecut  # Rydberg
    config.basis.ecut_rho = 8.0 * ecut
    config.scf.mixing.preconditioner = "none"
    config.scf.max_steps = 150
    config.scf.tol_energy = 1e-11
    config.scf.tol_density = 1e-10
    atoms.calc = Amaterasu(config, download=False)
    return atoms


def _free_energy(atoms) -> float:
    """The force-consistent energy in eV: with smearing, forces are the gradient of F."""
    atoms.get_potential_energy()
    return atoms.calc.scf.energies.free_energy * HARTREE_IN_EV


@pytest.fixture(scope="module")
def reference_cell():
    atoms = _make()
    return atoms, atoms.get_forces(), atoms.get_stress(voigt=False), atoms.get_volume()


def test_forces_sum_to_zero(reference_cell):
    """Translation invariance, limited here by the grid rather than by the formula.

    A norm-conserving run gets this to 8e-8 eV/A. PAW does not, and the reason is not the
    force expression -- which agrees with a finite difference of the energy to 2e-5 below
    -- but the energy itself: ``v_bar`` and the compensation charge are the sharpest things
    in the calculation, and the terms evaluated by summing over real-space grid points are
    correspondingly less translation-invariant. It is an egg-box error, so it decays with
    the cutoff, measured on this cell:

        40 Ry / 320 Ry   |sum F| = 1.98e-3 eV/A
        60 Ry / 480 Ry   |sum F| = 3.41e-4 eV/A

    The bound below is the 40 Ry value with room to spare, not a statement that 2e-3 is
    good; a production run wanting rigorous translation invariance needs a larger
    ``ecut_rho``, exactly as it does for the Pulay stress.
    """
    _, forces, _, _ = reference_cell
    assert np.abs(forces.sum(axis=0)).max() < 4e-3


@pytest.mark.parametrize(("atom", "axis"), [(0, 0), (0, 2), (2, 1)])
def test_force_against_a_self_consistent_finite_difference(reference_cell, atom, axis):
    _, forces, _, _ = reference_cell
    step = 0.004
    shift = np.zeros_like(POSITIONS)
    shift[atom, axis] = step
    difference = -(
        _free_energy(_make(displacement=shift)) - _free_energy(_make(displacement=-shift))
    ) / (2 * step)
    assert difference == pytest.approx(forces[atom, axis], abs=1e-3)


@pytest.mark.parametrize(("u", "v"), [(0, 0), (2, 2), (0, 1)])
def test_stress_against_a_self_consistent_finite_difference(reference_cell, u, v):
    _, _, stress, volume = reference_cell
    epsilon = 1e-3
    strain = np.zeros((3, 3))
    strain[u, v] += 0.5 * epsilon
    strain[v, u] += 0.5 * epsilon
    difference = (_free_energy(_make(strain=strain)) - _free_energy(_make(strain=-strain))) / (
        2 * epsilon * volume
    )
    # In GPa, so the tolerance means what it says about a pressure.
    # What is left here is Pulay stress, and it is worth saying how that was established
    # rather than assumed, because a constant offset between an analytic derivative and a
    # finite difference is exactly what a wrong derivative also looks like.
    #
    # Two signatures, both measured on this cell. It is anisotropic in the right way --
    # at 40 Ry the diagonal is off by 0.30-0.49 GPa and the shears by 0.001-0.024 GPa,
    # because Pulay stress comes from the set of plane waves inside the cutoff changing
    # under strain, which a volume-preserving shear barely does. And it decays with the
    # basis, which a wrong formula would not:
    #
    #                  40 Ry / 320 Ry   60 Ry / 480 Ry
    #     sigma[0,0]     0.329 GPa        0.032 GPa
    #     sigma[1,1]     0.488 GPa        0.028 GPa
    #     sigma[2,2]     0.299 GPa        0.032 GPa
    #     sigma[0,1]     0.001 GPa        0.003 GPa
    #     sigma[1,2]     0.024 GPa        0.007 GPa
    #
    # The worst component goes from 0.488 to 0.032 GPa for a 1.5x cutoff, while the forces
    # stay at 2-3e-5 eV/A throughout. A wrong strain derivative would not do that.
    assert difference * STRESS_TO_GPA == pytest.approx(stress[u, v] * STRESS_TO_GPA, abs=0.8)
