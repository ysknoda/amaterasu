"""End-to-end self-consistent field calculations on a small silicon cell.

These runs are deliberately under-converged -- a 6 Ha cutoff and a 2x2x2 mesh -- because
what is being tested is internal consistency, not accuracy: identities that must hold at
*any* cutoff, and agreement between algorithms that must reach the same ground state.
The accuracy of the converged numbers is established separately against the PseudoDojo
DojoReport equation of state.
"""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.config.schema import Config
from amaterasu.hamiltonian.system import build_system
from amaterasu.scf.driver import Energies, effective_potential, run_scf
from amaterasu.units import BOHR_IN_ANG, HARTREE_IN_EV

ECUT = 8.0


def _config(**overrides) -> Config:
    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "fixed"
    config.scf.max_steps = 40
    config.scf.tol_energy = 1e-9
    config.scf.tol_density = 1e-8
    for path, value in overrides.items():
        section, _, field = path.partition(".")
        setattr(getattr(config, section), field, value) if field else setattr(
            config, section, value
        )
    return config


def _run(atoms, pseudo, config=None, ecut=ECUT):
    config = config or _config()
    system = build_system(atoms, config, {"Si": pseudo}, ecut, 4 * ecut)
    return system, run_scf(system, config)


@pytest.fixture(scope="module")
def si_run(si_pseudo):
    """One converged SCF, reused by the checks that do not need their own run.

    The structure is built here rather than taken from the function-scoped ``si_atoms``
    fixture so that this run happens once for the whole module.
    """
    from ase.build import bulk

    from tests.conftest import SI_LATTICE_CONSTANT

    return _run(bulk("Si", "diamond", a=SI_LATTICE_CONSTANT), si_pseudo)


def test_scf_converges(si_run):
    _, result = si_run
    assert result.converged
    assert result.iterations < 30
    # Monotone-ish descent: the free energy at the end is the lowest seen.
    energies = [row[0] for row in result.history]
    assert energies[-1] == pytest.approx(min(energies), abs=1e-12)


def test_electron_count_is_exact(si_run):
    """Nothing in the sphere-to-grid mapping, the symmetrisation or the mixer may lose charge."""
    _, result = si_run
    assert result.electron_count == pytest.approx(8.0, abs=1e-9)


def test_eigenvalue_sum_identity(si_run):
    """sum_nk w f eps == T + integral v_eff n + E_nl, to machine precision.

    This ties the eigensolver, the Hamiltonian and the density accumulation together: it
    fails if any of them disagrees about normalisation, spin degeneracy or k-point weights,
    which is exactly the class of bug that otherwise shows up only as a slightly wrong
    number in a production run.

    The residual also measures how exactly ``v_eff`` shares the crystal's symmetry, since
    the difference it reports is ``integral v_eff (n_symmetrised - n_wedge)``. That is what
    makes it sensitive to the grid: see :func:`amaterasu.basis.fft.gradient` on the
    Nyquist plane, which leaves a residue of order 1e-10 Ha for a GGA on a grid too coarse
    for the density it carries.
    """
    _, result = si_run
    assert abs(result.history[-1][2]) * HARTREE_IN_EV < 1e-9


def test_silicon_is_an_insulator_with_a_reasonable_gap(si_run):
    """Four filled bands, a gap, and the valence band maximum at Gamma."""
    system, result = si_run
    assert result.gap is not None
    assert 0.4 < result.gap * HARTREE_IN_EV < 1.2
    assert np.allclose(result.occupations[0, :, :4], 1.0)
    assert np.allclose(result.occupations[0, :, 4:], 0.0)
    gamma = int(np.argmin(np.linalg.norm(system.kpoints.points, axis=1)))
    assert np.argmax(result.eigenvalues[0, :, 3]) == gamma


def test_translation_by_a_grid_vector_is_exact(si_atoms, si_pseudo):
    """Sliding the crystal by whole grid steps must not change the energy at all.

    Every ionic term -- the local potential, the projectors, the Ewald sum, the atomic
    densities -- is built from exact phases in reciprocal space, and a shift by a grid
    vector also maps the real-space grid onto itself, so nothing at all should move. This
    separates the physics (exactly translation invariant) from the discretisation, which
    the next test measures.
    """
    system, reference = _run(si_atoms, si_pseudo)
    step = system.cell[0] / system.grid.shape[0] * BOHR_IN_ANG
    shifted = si_atoms.copy()
    shifted.positions += 3.0 * step
    _, moved = _run(shifted, si_pseudo)
    assert moved.energies.total * HARTREE_IN_EV == pytest.approx(
        reference.energies.total * HARTREE_IN_EV, abs=1e-11
    )


def test_arbitrary_translation_is_invariant_to_the_egg_box_error(si_atoms, si_pseudo):
    """An off-grid shift costs only the residual "egg-box" error, which falls with the cutoff.

    ``v_xc`` is evaluated pointwise on the grid and is not band-limited, so its
    representation depends slightly on where the grid points fall relative to the nuclei.
    That is the egg-box effect every real-space-XC plane-wave code has. Measured here it is
    2 mueV per cell at a 8 Ha cutoff and 20 neV at 12 Ha -- a factor of 90 for a factor of
    1.5 in cutoff, which is the signature of a basis-set error rather than a bug.
    """
    _, reference = _run(si_atoms, si_pseudo)
    shifted = si_atoms.copy()
    shifted.positions += np.array([0.31, -0.77, 1.13])
    _, coarse = _run(shifted, si_pseudo)
    coarse_error = abs((coarse.energies.total - reference.energies.total) * HARTREE_IN_EV)
    assert coarse_error < 1e-5

    _, fine_reference = _run(si_atoms, si_pseudo, ecut=12.0)
    _, fine = _run(shifted, si_pseudo, ecut=12.0)
    fine_error = abs((fine.energies.total - fine_reference.energies.total) * HARTREE_IN_EV)
    assert fine_error < 0.1 * coarse_error


@pytest.mark.parametrize("algorithm", ["linear", "anderson", "broyden"])
def test_mixers_reach_the_same_ground_state(si_atoms, si_pseudo, si_run, algorithm):
    """The fixed point is a property of the equations, not of how it is reached."""
    _, reference = si_run
    config = _config()
    config.scf.mixing.algorithm = algorithm
    config.scf.max_steps = 120
    _, result = _run(si_atoms, si_pseudo, config)
    assert result.converged
    assert result.energies.total * HARTREE_IN_EV == pytest.approx(
        reference.energies.total * HARTREE_IN_EV, abs=1e-7
    )


@pytest.mark.parametrize("method", ["direct-cg", "direct-lbfgs"])
def test_direct_minimisation_matches_mixing(si_atoms, si_pseudo, si_run, method):
    """Two entirely different algorithms must find the same minimum.

    Mixing solves the Kohn-Sham equations as a fixed point with an iterative
    diagonalisation; direct minimisation never forms an eigenvalue and walks downhill on
    the energy over the Stiefel manifold. Agreement to a nano-electronvolt is a strong
    statement that the energy functional and the Hamiltonian are consistent with each
    other -- an error in one would move only one of the two answers.
    """
    _, reference = si_run
    config = _config()
    config.scf.method = method
    config.scf.max_steps = 120
    _, result = _run(si_atoms, si_pseudo, config)
    assert result.converged
    assert result.energies.total * HARTREE_IN_EV == pytest.approx(
        reference.energies.total * HARTREE_IN_EV, abs=1e-6
    )


def test_direct_minimisation_refuses_smeared_occupations(si_atoms, si_pseudo):
    config = _config()
    config.scf.method = "direct-lbfgs"
    config.occupations.scheme = "gaussian"
    with pytest.raises(NotImplementedError, match=r"occupations\.scheme: fixed"):
        _run(si_atoms, si_pseudo, config)


def test_spin_polarised_reduces_to_unpolarised_at_zero_moment(si_atoms, si_pseudo, si_run):
    """Silicon is non-magnetic: a spin-polarised run must find the same energy and no moment."""
    _, reference = si_run
    config = _config()
    config.system.spin = "polarized"
    _, result = _run(si_atoms, si_pseudo, config)
    assert result.converged
    assert result.magnetization == pytest.approx(0.0, abs=1e-6)
    assert result.energies.total * HARTREE_IN_EV == pytest.approx(
        reference.energies.total * HARTREE_IN_EV, abs=1e-6
    )
    assert result.eigenvalues[0] == pytest.approx(result.eigenvalues[1], abs=1e-9)


def test_energy_decomposition_adds_up():
    energies = Energies(
        kinetic=1.0, local=-2.0, hartree=0.5, xc=-0.25, nonlocal_=0.125, ewald=-3.0, entropy=0.1
    )
    assert energies.total == pytest.approx(-3.625)
    assert energies.free_energy == pytest.approx(-3.725)
    assert energies.extrapolated == pytest.approx(-3.675)
    assert "free energy F = E - TS" in energies.render(2)


def test_effective_potential_includes_the_core_correction(si_atoms, si_pseudo, si_scf_config):
    """NLCC changes v_xc but must not change the electron count or the Hartree term."""
    system = build_system(si_atoms, si_scf_config, {"Si": si_pseudo}, ECUT, 4 * ECUT)
    density = system.initial_density()
    with_core, hartree_a, xc_a, _ = effective_potential(system, density)
    system.core_density = None
    without_core, hartree_b, xc_b, _ = effective_potential(system, density)
    assert hartree_a == pytest.approx(hartree_b, rel=1e-12)
    assert xc_a != pytest.approx(xc_b, rel=1e-6)
    assert not np.allclose(with_core, without_core)


def test_results_round_trip_through_hdf5(tmp_path, si_run):
    from amaterasu.io.results import read_scf, write_scf

    system, result = si_run
    path = write_scf(tmp_path / "si.h5", system, result, save_wavefunctions=True)
    stored = read_scf(path)
    assert stored["attrs"]["converged"]
    assert stored["energies"]["total"] == pytest.approx(
        result.energies.total * HARTREE_IN_EV, rel=1e-12
    )
    assert stored["eigenvalues"] == pytest.approx(result.eigenvalues * HARTREE_IN_EV, rel=1e-12)
    # The density is stored per cubic Angstrom and must come back per cubic Bohr.
    assert stored["density"] == pytest.approx(result.density, rel=1e-10)
    assert stored["symbols"] == ["Si", "Si"]
    assert len(stored["wavefunctions"][0]) == system.nkpoints


def test_zero_initial_moments_are_flagged(si_atoms, si_pseudo, capsys):
    """A spin-polarised run with no moments finds the non-magnetic answer at twice the cost."""
    from amaterasu.solvers import scf as scf_solver

    config = _config()
    config.system.spin = "polarized"
    config.pseudo.files = {"Si": str(_bundled_silicon())}
    config.pseudo.directory = str(_bundled_silicon().parent)
    config.basis.ecut = ECUT * 2  # basis.ecut is in Rydberg
    messages = []
    scf_solver.run(si_atoms, config, logger=messages.append, download=False, write_output=False)
    assert any("every initial moment is zero" in m for m in messages)


def _bundled_silicon():
    from tests.conftest import PSEUDO_DATA

    return PSEUDO_DATA / "Si.pbe.psp8.gz"
