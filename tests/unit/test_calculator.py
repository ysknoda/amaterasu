"""The ASE calculator: units at the boundary, and the restart that makes trajectories cheap."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from amaterasu.calculator import Amaterasu
from amaterasu.config.schema import Config
from amaterasu.units import BOHR_IN_ANG, HARTREE_IN_EV, HARTREE_PER_BOHR3_IN_GPA

from ..conftest import PSEUDO_DATA


def _config(**overrides) -> Config:
    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "fixed"
    config.pseudo.directory = str(PSEUDO_DATA)
    config.pseudo.files = {"Si": "Si.pbe.psp8.gz"}
    config.basis.ecut = 16.0  # Rydberg, i.e. 8 Ha
    config.scf.max_steps = 60
    config.scf.tol_energy = 1e-9
    config.scf.tol_density = 1e-8
    for path, value in overrides.items():
        section, _, field = path.partition(".")
        setattr(getattr(config, section), field, value)
    return config


@pytest.fixture
def silicon(si_atoms):
    atoms = si_atoms.copy()
    atoms.calc = Amaterasu(_config(), download=False)
    return atoms


def test_energy_forces_and_stress_are_in_ase_units(silicon):
    energy = silicon.get_potential_energy()
    forces = silicon.get_forces()
    stress = silicon.get_stress()

    # Silicon's total energy is about -115 eV per atom with this dataset; a factor of 27
    # out would mean a missing Hartree-to-eV conversion.
    assert -260.0 < energy < -200.0
    assert forces.shape == (2, 3)
    assert np.abs(forces).max() < 1e-10  # ideal diamond: zero by symmetry
    assert stress.shape == (6,)
    # Voigt shear components vanish for a cubic crystal, and the diagonal is uniform.
    assert stress[3:] == pytest.approx(np.zeros(3), abs=1e-12)
    assert stress[:3] == pytest.approx(np.full(3, stress[0]), abs=1e-12)


def test_free_energy_and_extrapolated_energy_differ_only_by_the_entropy(si_atoms):
    """With smearing the three energies are distinct, and the forces belong to the free one."""
    atoms = si_atoms.copy()
    atoms.calc = Amaterasu(
        _config(**{"occupations.scheme": "gaussian", "occupations.smearing": 0.5}),
        download=False,
    )
    atoms.get_potential_energy()
    energies = atoms.calc.scf.energies
    assert energies.free_energy < energies.total
    assert energies.extrapolated == pytest.approx(
        0.5 * (energies.total + energies.free_energy), rel=1e-12
    )
    assert atoms.calc.results["energy"] == pytest.approx(
        energies.extrapolated * HARTREE_IN_EV, rel=1e-12
    )
    assert atoms.calc.results["free_energy"] == pytest.approx(
        energies.free_energy * HARTREE_IN_EV, rel=1e-12
    )


def test_pressure_is_reported_in_gigapascals(silicon):
    pressure = silicon.calc.get_pressure(silicon)
    stress = silicon.get_stress()
    expected = -np.mean(stress[:3]) / (HARTREE_IN_EV / BOHR_IN_ANG**3) * HARTREE_PER_BOHR3_IN_GPA
    assert pressure == pytest.approx(expected, rel=1e-12)
    assert abs(pressure) < 200.0


def test_restart_from_the_previous_geometry_saves_iterations(si_atoms):
    """A trajectory step should not start from the superposed atomic densities again."""
    atoms = si_atoms.copy()
    atoms.calc = Amaterasu(_config(), download=False)
    atoms.get_potential_energy()
    first = atoms.calc.iteration_history[-1]

    atoms.positions[1] += np.array([0.01, 0.0, 0.0])
    atoms.get_potential_energy()
    second = atoms.calc.iteration_history[-1]
    assert second < first
    assert len(atoms.calc.iteration_history) == 2


def test_changing_the_cell_discards_the_stored_orbitals(si_atoms):
    """A different cell is a different basis, so the stored coefficients are meaningless."""
    atoms = si_atoms.copy()
    atoms.calc = Amaterasu(_config(), download=False)
    atoms.get_potential_energy()
    baseline = atoms.calc.iteration_history[-1]

    atoms.set_cell(atoms.cell * 1.05, scale_atoms=True)
    atoms.get_potential_energy()
    assert atoms.calc.iteration_history[-1] >= baseline - 1


def test_spin_polarised_run_reports_a_moment(si_atoms):
    atoms = si_atoms.copy()
    atoms.set_initial_magnetic_moments([0.5, -0.5])
    atoms.calc = Amaterasu(
        _config(**{"system.spin": "polarized", "occupations.scheme": "gaussian"}), download=False
    )
    atoms.get_potential_energy()
    # Silicon is not magnetic: whatever moment it is started with must collapse.
    assert abs(atoms.calc.results["magmom"]) < 1e-3


def test_calculator_drives_an_ase_optimiser(si_atoms):
    """The point of the calculator: ASE's own algorithms work without any adapter."""
    from ase.optimize import FIRE

    atoms = si_atoms.copy()
    atoms.positions[1] += np.array([0.08, 0.0, 0.0])
    atoms.calc = Amaterasu(_config(), download=False)
    before = np.abs(atoms.get_forces()).max()
    FIRE(atoms, logfile=None).run(fmax=0.05, steps=3)
    after = np.abs(atoms.get_forces()).max()
    assert after < before


def test_non_periodic_input_is_rejected(si_atoms):
    molecule = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=8.0 * np.eye(3), pbc=False)
    molecule.calc = Amaterasu(_config(), download=False)
    with pytest.raises(ValueError, match="3D-periodic"):
        molecule.get_potential_energy()
