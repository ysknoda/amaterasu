"""The non-self-consistent band-structure solver.

The decisive test here is not that a picture comes out. It is that a k-point which appears
in *both* the self-consistent mesh and the band path gets the same eigenvalues either way.
That single comparison exercises everything the solver does differently from the SCF: the
system is rebuilt for new k-points, the converged density is carried across, the effective
potential is reassembled from it, and (for PAW) the occupancy matrices and their coupling
are recomputed from wavefunctions belonging to the *other* k-point set. If any of those is
subtly wrong, the two numbers stop agreeing.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from amaterasu.config.schema import Config
from amaterasu.hamiltonian.system import build_system
from amaterasu.io.results import read_bands, write_bands
from amaterasu.scf.driver import run_scf
from amaterasu.solvers import bands as bands_solver
from amaterasu.solvers.scf import SCFRun
from amaterasu.system.kpoints import band_path_with_cell
from amaterasu.units import HARTREE_IN_EV


def _config():
    config = Config(structure="-", solver="bands")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "fixed"
    config.scf.max_steps = 40
    config.scf.tol_energy = 1e-9
    config.scf.tol_density = 1e-8
    config.bands.npoints = 12
    return config


@pytest.fixture(scope="module")
def converged(si_atoms_module, si_pseudo):
    config = _config()
    system = build_system(si_atoms_module, config, {"Si": si_pseudo}, 10.0, 40.0)
    result = run_scf(system, config)
    assert result.converged
    return SCFRun(system=system, result=result, pseudopotential_paths={})


@pytest.fixture(scope="module")
def si_atoms_module():
    from ase.build import bulk

    return bulk("Si", "diamond", a=5.43)


def test_a_shared_kpoint_gets_the_same_eigenvalues_either_way(converged):
    """The whole point of the module, in one comparison.

    The k-points of the self-consistent mesh are fed back in as if they were a path. Every
    eigenvalue has to come back unchanged: same density, same potential, same operator --
    only the bookkeeping around it differs.
    """
    config = _config()
    mesh = np.asarray(converged.system.kpoints.points)
    again = bands_solver.solve_along_path(converged, config, mesh, logger=lambda *_: None)
    assert again == pytest.approx(converged.result.eigenvalues, abs=1e-9)


def test_the_path_solver_does_not_disturb_the_scf_run(converged):
    """It reuses the converged state, so it must not modify it."""
    before = converged.result.eigenvalues.copy()
    density = converged.result.density.copy()
    bands_solver.solve_along_path(
        converged, _config(), np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), logger=lambda *_: None
    )
    assert converged.result.eigenvalues == pytest.approx(before, abs=0.0)
    assert converged.result.density == pytest.approx(density, abs=0.0)


def test_gamma_has_the_degeneracies_the_diamond_group_requires(converged):
    """Silicon's valence band top at Gamma is the threefold Gamma_25'.

    A path calculation is where a broken symmetrisation or a mis-set k-point would show up
    as a split degeneracy, because nothing averages it away.
    """
    eigenvalues = bands_solver.solve_along_path(
        converged, _config(), np.zeros((1, 3)), logger=lambda *_: None
    )[0, 0]
    triplet = eigenvalues[1:4] * HARTREE_IN_EV
    assert triplet.max() - triplet.min() < 1e-6, triplet
    assert eigenvalues[0] < eigenvalues[1] - 0.1


def test_explicit_kpoints_are_not_symmetry_reduced():
    """Folding a path into the irreducible wedge would quietly delete most of it."""
    config = _config()
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.1, 0.0]])
    band_config = bands_solver._explicit_kpoint_config(config, points)
    assert band_config.kpoints.scheme == "explicit"
    assert band_config.kpoints.symmetry is False
    assert band_config.kpoints.weights is None
    assert np.asarray(band_config.kpoints.points) == pytest.approx(points)
    # the original is untouched
    assert config.kpoints.scheme == "monkhorst-pack"


def test_an_explicit_path_keeps_the_input_cell(si_atoms_module):
    """Only ``auto`` goes through seekpath, so only ``auto`` can restandardise the cell."""
    path, cell = band_path_with_cell(si_atoms_module, "GXG", npoints=8)
    assert cell is si_atoms_module
    assert len(path.kpts) == 8


def test_auto_reports_the_cell_its_labels_belong_to(si_atoms_module):
    path, cell = band_path_with_cell(si_atoms_module, "auto", npoints=8)
    assert set("GXLW") <= set(path.special_points)
    assert cell.get_chemical_formula() == si_atoms_module.get_chemical_formula()
    # seekpath answers in its own primitive cell, whose volume must still be the crystal's
    assert cell.get_volume() == pytest.approx(si_atoms_module.get_volume(), rel=1e-9)


def test_band_structure_round_trips_through_hdf5(tmp_path, converged, si_atoms_module):
    path, _ = band_path_with_cell(si_atoms_module, "GXG", npoints=6)
    eigenvalues = bands_solver.solve_along_path(
        converged, _config(), path.kpts, logger=lambda *_: None
    )
    structure = bands_solver.BandStructure(
        path=path,
        eigenvalues=eigenvalues,
        fermi_level=converged.result.fermi_level,
        scf=converged,
    )
    target = write_bands(tmp_path / "bands.h5", structure)
    back = read_bands(target)
    assert back["attrs"]["solver"] == "bands"
    assert back["eigenvalues"] == pytest.approx(eigenvalues * HARTREE_IN_EV, rel=1e-12)
    assert back["fermi_level"] == pytest.approx(converged.result.fermi_level * HARTREE_IN_EV)
    assert back["kpoints"] == pytest.approx(np.asarray(path.kpts))
    assert "G" in back["special_point_labels"]
    assert len(back["distance"]) == len(path.kpts)


def test_the_reported_gap_is_the_one_on_the_path(converged, si_atoms_module):
    path, _ = band_path_with_cell(si_atoms_module, "GXG", npoints=6)
    eigenvalues = bands_solver.solve_along_path(
        converged, _config(), path.kpts, logger=lambda *_: None
    )
    structure = bands_solver.BandStructure(
        path=path,
        eigenvalues=eigenvalues,
        fermi_level=converged.result.fermi_level,
        scf=converged,
    )
    gap = structure.gap
    assert gap is not None
    below = eigenvalues[eigenvalues <= structure.fermi_level].max()
    above = eigenvalues[eigenvalues > structure.fermi_level].min()
    assert gap == pytest.approx(above - below)
    assert "lower bound" in structure.render()


def test_a_grid_mismatch_is_refused(converged):
    """Reusing a density across different FFT grids would be silently wrong."""
    other = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.eye(3) * 12.0,
        pbc=True,
    )
    mismatched = build_system(other, _config(), converged.system.pseudopotentials, 10.0, 40.0)
    with pytest.raises(ValueError, match="density cannot be reused"):
        bands_solver._check_reused_potential(converged.system, mismatched)
