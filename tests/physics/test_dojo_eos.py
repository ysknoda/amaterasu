"""Phase 2 gate 5: reproduce the PseudoDojo DojoReport equation of state.

The DojoReport records ABINIT total energies for a set of volumes, computed with exactly
the psp8 file bundled under ``tests/data/pseudos/`` at a stated cutoff. Running the same
volumes here therefore compares two independent plane-wave implementations with the
pseudopotential, the functional and the structure all held fixed -- there is nothing left
to differ except the machinery being tested.

Two things are checked, and the distinction matters:

* the **shape** of ``E(V)`` -- the fitted ``V0``, ``B0``, ``B1`` and the delta-gauge
  between the two curves. These are what physics depends on, and they must agree closely.
* the **offset** between the two total energies. A constant shift is expected: the
  reference used a denser k-mesh than is affordable here and ABINIT applies its
  ``ecutsm`` cutoff smoothing, neither of which changes a difference of energies. What
  must be small is the *variation* of the offset across the volume range, which is the
  part that would distort the equation of state.

Marked ``physics`` because it runs seven self-consistent calculations; it is excluded
from the default suite and run deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from scipy.optimize import least_squares

from amaterasu.config.schema import Config
from amaterasu.hamiltonian.system import build_system
from amaterasu.pseudo.readers import read_pseudopotential
from amaterasu.scf.driver import run_scf
from amaterasu.units import BOHR_IN_ANG, HARTREE_IN_EV, HARTREE_PER_BOHR3_IN_GPA

pytestmark = [pytest.mark.physics, pytest.mark.slow]

DATA = Path(__file__).resolve().parents[1] / "data"
EV_PER_A3_IN_GPA = HARTREE_PER_BOHR3_IN_GPA / (HARTREE_IN_EV / BOHR_IN_ANG**3)


def birch_murnaghan(volume, e0, v0, b0, b1):
    """Third-order Birch-Murnaghan energy, with ``b0`` in the same units as ``e0``/volume."""
    eta = (v0 / volume) ** (2.0 / 3.0) - 1.0
    return e0 + 9.0 * v0 * b0 / 16.0 * (eta**3 * b1 + eta**2 * (6.0 - 4.0 * (eta + 1.0)))


def fit_eos(volumes, energies):
    volumes, energies = np.asarray(volumes, float), np.asarray(energies, float)
    guess = [energies.min(), volumes[int(np.argmin(energies))], 0.5, 4.0]
    result = least_squares(
        lambda p: birch_murnaghan(volumes, *p) - energies, guess, xtol=1e-15, ftol=1e-15
    )
    return result.x, float(np.sqrt(np.mean(result.fun**2)))


def delta_gauge(a, b, npoints=2001):
    """Lejaeghere delta between two fitted equations of state, in meV/atom.

    Both curves are shifted to their own minimum before being compared over +-6% of the
    mean equilibrium volume, so the measure is insensitive to any constant energy offset.
    """
    v0 = 0.5 * (a[1] + b[1])
    grid = np.linspace(0.94 * v0, 1.06 * v0, npoints)
    curve_a = birch_murnaghan(grid, 0.0, *a[1:]) - birch_murnaghan(np.array([a[1]]), 0.0, *a[1:])
    curve_b = birch_murnaghan(grid, 0.0, *b[1:]) - birch_murnaghan(np.array([b[1]]), 0.0, *b[1:])
    return 1000.0 * float(
        np.sqrt(np.trapezoid((curve_a - curve_b) ** 2, grid) / (grid[-1] - grid[0]))
    )


@pytest.fixture(scope="module")
def silicon_eos():
    """amaterasu total energies at the DojoReport volumes for silicon."""
    reference = json.loads((DATA / "dojo" / "deltafactor.json").read_text())["Si"]
    pseudo = read_pseudopotential(DATA / "pseudos" / "Si.pbe.psp8.gz")

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [8, 8, 8]
    config.occupations.scheme = "fixed"
    config.scf.max_steps = 60
    config.scf.tol_energy = 1e-9
    config.scf.tol_density = 1e-9

    template = bulk("Si", "diamond")
    natoms = len(template)
    volumes = np.asarray(reference["volumes_A3"], float) / reference["num_sites"]
    energies = []
    for volume in volumes:
        scale = (volume * natoms / template.get_volume()) ** (1 / 3)
        atoms = template.copy()
        atoms.set_cell(template.cell * scale, scale_atoms=True)
        system = build_system(
            atoms, config, {"Si": pseudo}, reference["ecut_Ha"], 4 * reference["ecut_Ha"]
        )
        result = run_scf(system, config)
        assert result.converged
        energies.append(result.energies.extrapolated * HARTREE_IN_EV / natoms)
    return reference, volumes, np.asarray(energies)


def test_equation_of_state_matches_the_dojo_report(silicon_eos):
    """V0, B0 and B1 must agree with the published fit to a fraction of a percent."""
    reference, volumes, energies = silicon_eos
    mine, residual = fit_eos(volumes, energies)
    assert residual * 1000 < 0.05  # a clean Birch-Murnaghan fit, in meV/atom

    assert mine[1] == pytest.approx(reference["v0_A3_per_atom"], rel=2e-3)
    assert mine[2] * EV_PER_A3_IN_GPA == pytest.approx(reference["b0_GPa"], rel=0.02)
    assert mine[3] == pytest.approx(reference["b1"], rel=0.05)

    # The PBE lattice constant of silicon. Diamond has *eight* atoms in the conventional
    # cubic cell, not the four of plain fcc, so a0 = (8 V0)^(1/3). The result is about
    # 0.7% above the experimental 5.431 A, the usual PBE overestimate for a covalent solid.
    lattice_constant = (8.0 * mine[1]) ** (1 / 3)
    assert 5.46 < lattice_constant < 5.48


def test_delta_gauge_against_the_same_dataset_is_negligible(silicon_eos):
    """The two curves must be closer to each other than the dataset is to all-electron.

    If the delta between amaterasu and ABINIT with the same pseudopotential is small
    compared with the dataset's published delta against WIEN2k, then amaterasu inherits
    that published figure -- which is the chain of reasoning the Phase 4 PAW gate rests on.
    """
    reference, volumes, energies = silicon_eos
    mine, _ = fit_eos(volumes, energies)
    # Both curves must be per atom: the report stores energies for the whole cell while
    # the volumes above are already divided by the site count.
    per_atom = np.asarray(reference["etotals_eV"], float) / reference["num_sites"]
    theirs, _ = fit_eos(volumes, per_atom)
    delta = delta_gauge(mine, theirs)
    assert delta < 0.5 * reference["dfact_meV"] or delta < 0.1


def test_energy_offset_is_constant_across_the_volume_range(silicon_eos):
    """A rigid shift is harmless; a volume-dependent one would distort the pressure.

    The offset comes from the reference's denser k-mesh and ABINIT's ``ecutsm`` smoothing.
    Neither survives a difference of energies, so what is required is that the spread be
    far smaller than the offset itself.
    """
    reference, _, energies = silicon_eos
    difference = energies - np.asarray(reference["etotals_eV"]) / reference["num_sites"]
    assert abs(float(np.mean(difference))) * 1000 < 5.0
    assert float(np.std(difference)) * 1000 < 0.1


def test_the_bundled_dataset_is_the_one_the_report_used():
    """The comparison is only meaningful if the psp8 file is byte-identical."""
    import gzip
    import hashlib

    reference = json.loads((DATA / "dojo" / "deltafactor.json").read_text())["Si"]
    with gzip.open(DATA / "pseudos" / "Si.pbe.psp8.gz", "rb") as handle:
        digest = hashlib.md5(handle.read()).hexdigest()
    assert digest == reference["md5"]
