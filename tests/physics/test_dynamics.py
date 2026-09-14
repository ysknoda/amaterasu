"""Phase 3 gates 4 and 5: energy conservation in NVE, and symmetry through a relaxation."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from amaterasu.config.schema import Config
from amaterasu.solvers import md as md_solver
from amaterasu.solvers import relax as relax_solver
from amaterasu.system.symmetry import analyze

from ..conftest import PSEUDO_DATA

pytestmark = [pytest.mark.physics, pytest.mark.slow]


def _config(solver: str) -> Config:
    config = Config(structure="-", solver=solver)
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "gaussian"
    config.occupations.smearing = 0.1
    config.pseudo.directory = str(PSEUDO_DATA)
    config.pseudo.files = {"Si": "Si.pbe.psp8.gz"}
    config.basis.ecut = 16.0  # Rydberg
    config.scf.tol_energy = 1e-10
    config.scf.tol_density = 1e-9
    config.scf.max_steps = 80
    return config


def test_nve_conserves_energy(tmp_path):
    """Gate 4: the conserved energy stays put, and does not drift.

    NVE is the sharpest available test of a force implementation, because the answer is
    known exactly: nothing should change. A force that is not the gradient of the energy
    pumps energy in or out steadily. The conserved quantity is the *free* energy plus the
    kinetic energy, since the force is the gradient of F once the occupations are smeared.

    The run is 150 fs, which covers a little over two periods of silicon's 15.5 THz optical
    mode. That matters: a linear drift fitted over less than one period measures the slope
    of the oscillation rather than any drift, and swings sign as the window changes. The
    peak-to-peak spread is the quantity that means something at any length, so it is
    checked first and tightly.
    """
    config = _config("md")
    config.md.ensemble = "nve"
    config.md.timestep = 1.0
    config.md.steps = 150
    config.md.temperature = 400.0
    config.md.seed = 7
    config.output.directory = str(tmp_path)

    atoms = bulk("Si", "diamond", a=5.43)
    result = md_solver.run(atoms, config, logger=lambda *_: None, download=False)
    assert result.steps == 150
    spread = result.energy_spread(len(atoms))
    assert spread < 0.5, f"conserved energy varies by {spread:.4f} meV/atom"
    drift = result.energy_drift(len(atoms))
    assert abs(drift) < 1.0, f"energy drift {drift:.4f} meV/atom/ps"
    # The temperature must actually be doing something, or the test is vacuous.
    assert result.temperatures.mean() > 50.0


def test_relaxation_restores_the_ideal_lattice(tmp_path):
    """Gate 5: relaxing a displaced atom puts it back and keeps the space group.

    Diamond's atomic positions are fixed by symmetry, so the exact answer is known: the
    displaced atom must return to its Wyckoff site and the space group must come back to
    Fd-3m (227). Anything else means the forces point somewhere other than downhill.
    """
    config = _config("relax")
    config.relax.algorithm = "fire"
    config.relax.fmax = 0.01
    config.relax.max_steps = 40
    config.relax.fix_symmetry = False
    config.output.directory = str(tmp_path)

    ideal = bulk("Si", "diamond", a=5.43)
    atoms = ideal.copy()
    atoms.positions[1] += np.array([0.08, -0.05, 0.03])
    assert analyze(atoms, symprec=1e-4).number != 227

    result = relax_solver.run(atoms, config, logger=lambda *_: None, download=False)
    assert result.converged
    assert result.final_energy < result.initial_energy
    assert analyze(result.atoms, symprec=1e-3).number == 227

    # The bond length must come back to the ideal (sqrt(3)/4) a.
    distance = result.atoms.get_distance(0, 1, mic=True)
    assert distance == pytest.approx(np.sqrt(3) / 4 * 5.43, abs=0.01)
