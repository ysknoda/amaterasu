"""PAW on the plane-wave side: compensation charge, overlap operator, symmetrisation.

Everything here runs on the synthetic dataset built in ``conftest``, so the tests are fast
and depend on no downloaded table. What they check are the identities that hold for *any*
PAW calculation, and each one fails loudly for a different mistake:

* the compensated density is neutral -- without which the plane-wave Hartree term could
  not stand in for the whole of the electrostatics;
* the smooth orbitals are normalised in the ``S`` metric, and the charge they are missing
  is exactly the augmentation charge;
* the eigenvalue sum matches the Hamiltonian's own decomposition, which ties the
  generalised eigensolver, the one-centre coupling and the density accumulation together;
* a symmetry-reduced Brillouin-zone sum gives the same answer as the full zone.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from amaterasu.config import load_config_mapping
from amaterasu.hamiltonian.paw import harmonic_rotation, make_setups
from amaterasu.hamiltonian.system import build_system
from amaterasu.scf.driver import compute_occupancies, effective_potential, paw_state, run_scf
from amaterasu.system.symmetry import analyze, cartesian_operations
from amaterasu.units import ANG_IN_BOHR
from amaterasu.xc.driver import XCFunctional


def _atoms():
    """Two synthetic atoms in a body-centred arrangement, low enough symmetry to be honest."""
    return Atoms(
        "X2", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 4.2, pbc=True
    )


def _config(symmetry=True, **overrides):
    settings = {
        "structure": "x.str",
        "basis": {"ecut": "24 Ry"},
        "kpoints": {"grid": [2, 2, 2], "symmetry": symmetry},
        "occupations": {"scheme": "gaussian", "smearing": 0.2},
        "scf": {"max_steps": 40, "tol_energy": 1.0e-8, "tol_density": 1.0e-7},
        "xc": {"functional": "LDA-PW"},
    }
    settings.update(overrides)
    return load_config_mapping(settings)


@pytest.fixture(scope="module")
def paw_system(synthetic_paw):
    atoms = _atoms()
    return build_system(atoms, _config(), {"X": synthetic_paw}, ecut=12.0, ecut_rho=96.0)


@pytest.fixture(scope="module")
def paw_result(paw_system):
    return run_scf(paw_system, _config())


def test_system_is_recognised_as_paw(paw_system):
    assert paw_system.is_paw
    assert paw_system.ewald == 0.0, "PAW puts the nuclei inside the compensation charge"
    assert paw_system.paw.n_projectors == 2 * 5


def test_mixed_datasets_are_refused(synthetic_paw, si_atoms):
    """The ionic energy is partitioned differently; there is no consistent middle."""
    from amaterasu.pseudo.readers import read_pseudopotential

    from ..conftest import PSEUDO_DATA

    silicon = read_pseudopotential(PSEUDO_DATA / "Si.pbe.psp8.gz")
    atoms = Atoms(
        "XSi", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 4.2, pbc=True
    )
    with pytest.raises(NotImplementedError, match="mixes"):
        build_system(atoms, _config(), {"X": synthetic_paw, "Si": silicon}, ecut=8.0, ecut_rho=64.0)


def test_direct_minimisation_is_refused(paw_system):
    """It minimises over the Euclidean Stiefel manifold; PAW's is the S metric."""
    config = _config()
    config.scf.method = "direct-lbfgs"
    config.occupations.scheme = "fixed"
    with pytest.raises(NotImplementedError, match="S metric"):
        run_scf(paw_system, config)


def test_compensated_density_is_neutral(paw_system, paw_result):
    """``n~ + n~_c + Z~`` integrates to zero, which is what removes the Ewald sum."""
    occupancies = compute_occupancies(paw_system, paw_result.wavefunctions, paw_result.occupations)
    state = paw_state(paw_system, occupancies)
    volume_element = paw_system.grid.cell_volume_per_point
    total = (
        float(np.sum(paw_result.density))
        + float(np.sum(paw_system.core_density))
        + float(np.sum(state.compensation))
    ) * volume_element
    assert abs(total) < 1e-7


def test_smooth_charge_is_short_by_the_augmentation_charge(paw_system, paw_result):
    """``integral |psi~|^2 = 1 - sum_ij D_ij O_ij`` -- the overlap operator's whole content."""
    occupancies = compute_occupancies(paw_system, paw_result.wavefunctions, paw_result.occupations)
    smooth = float(np.sum(paw_result.density)) * paw_system.grid.cell_volume_per_point
    augmented = float(np.einsum("sij,ij->", occupancies, paw_system.paw.overlap))
    assert paw_system.nelectrons - smooth == pytest.approx(augmented, abs=1e-8)


def test_starting_density_carries_the_smooth_charge(paw_system):
    """Kerker freezes G = 0, so the starting charge has to be the right one from the outset."""
    density = paw_system.initial_density()
    charge = float(np.sum(density)) * paw_system.grid.cell_volume_per_point
    assert charge == pytest.approx(paw_system.smooth_electron_count(), rel=1e-9)
    # The sign of the difference is a property of the dataset -- the augmentation charge
    # can go either way -- but it is never zero, which is the whole point.
    assert abs(charge - paw_system.nelectrons) > 1e-3


def test_orbitals_are_orthonormal_in_the_s_metric(paw_system, paw_result):
    from amaterasu.hamiltonian.operator import Hamiltonian

    potential, *_ = effective_potential(paw_system, paw_result.density, None)
    hamiltonian = Hamiltonian(paw_system, potential)
    for ik in range(paw_system.nkpoints):
        block = paw_result.wavefunctions[0][ik]
        overlap = block.conj() @ hamiltonian.apply_overlap(block, ik).T
        assert overlap == pytest.approx(np.eye(len(block)), abs=1e-8)


def test_eigenvalue_sum_identity(paw_system, paw_result):
    """``sum w f eps == T + int v_eff n~ + sum D_ij H_ij``.

    The tightest single check on a PAW iteration: it fails if the generalised eigensolver,
    the one-centre coupling and the density accumulation disagree about anything at all.
    """
    assert abs(paw_result.history[-1][2]) < 1e-9


def test_energy_decomposition_has_no_ewald_term(paw_result):
    assert paw_result.energies.ewald == 0.0
    assert paw_result.energies.nonlocal_ != 0.0  # the one-centre correction lives here


def test_symmetrised_and_full_zone_agree(synthetic_paw):
    """The decisive check on the occupancy symmetrisation.

    A wedge sum gives a ``D^a_ij`` with the symmetry of the wedge, not of the crystal, and
    the one-centre energy is not a scalar function of it: an occupancy that should be
    spherical picks up a quadrupole, and with it a spurious compensation charge. If the
    rotation matrices or the atom mapping were wrong, this is where it would show.
    """
    atoms = _atoms()
    energies = []
    for symmetry in (True, False):
        config = _config(symmetry=symmetry)
        system = build_system(atoms, config, {"X": synthetic_paw}, ecut=10.0, ecut_rho=80.0)
        energies.append(run_scf(system, config).energies.free_energy)
    assert energies[0] == pytest.approx(energies[1], abs=1e-8)


# ---------------------------------------------------------------- rotation matrices
def test_harmonic_rotations_are_orthogonal_and_defining(si_atoms):
    """``Y_L(R n) = sum_L' W_LL' Y_L'(n)`` for every operation of the space group."""
    from amaterasu.pseudo.paw.angular import real_harmonics

    symmetry = analyze(si_atoms)
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    operations = cartesian_operations(symmetry, cell)
    assert len(operations) == 48

    rng = np.random.default_rng(0)
    directions = rng.normal(size=(7, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for matrix in operations:
        rotation = harmonic_rotation(3, matrix)
        assert rotation @ rotation.T == pytest.approx(np.eye(16), abs=1e-12)
        assert real_harmonics(3, directions @ matrix) == pytest.approx(
            rotation @ real_harmonics(3, directions), abs=1e-12
        )


def test_symmetrising_a_spherical_occupancy_changes_nothing(synthetic_paw, paw_system):
    setups = make_setups({"X": synthetic_paw}, XCFunctional("LDA-PW"))
    single = setups["X"].atomic_occupancies(1)
    occupancies = paw_system.paw.scatter([single, single], 1)
    assert paw_system.paw.symmetrize(occupancies) == pytest.approx(occupancies, abs=1e-12)


def test_symmetrisation_is_idempotent(paw_system, paw_result):
    occupancies = compute_occupancies(paw_system, paw_result.wavefunctions, paw_result.occupations)
    once = paw_system.paw.symmetrize(occupancies)
    assert paw_system.paw.symmetrize(once) == pytest.approx(once, abs=1e-12)


def test_symmetrisation_preserves_the_augmentation_charge(paw_system, paw_result):
    """The trace against O is a scalar, so an average over the group cannot move it."""
    occupancies = compute_occupancies(paw_system, paw_result.wavefunctions, paw_result.occupations)
    overlap = paw_system.paw.overlap
    before = float(np.einsum("sij,ij->", occupancies, overlap))
    after = float(np.einsum("sij,ij->", paw_system.paw.symmetrize(occupancies), overlap))
    assert after == pytest.approx(before, rel=1e-12)


# ---------------------------------------------------- compensation-charge diagnostics
def test_an_over_concentrated_compensation_charge_is_warned_about(synthetic_paw):
    """The energy functional stops being bounded below before anything else complains.

    A collapsed run converges -- to a total energy hundreds of eV too low. Nothing in the
    SCF notices, so the check has to happen when the machinery is built, from the shape
    alone. See :func:`amaterasu.pseudo.paw.shape.equivalent_alpha_rc2` for the measurements
    the threshold sits between.
    """
    config = _config(
        pseudo={"method": "paw", "compensation_shape": "bessel", "compensation_scale": 0.6}
    )
    system = build_system(_atoms(), config, {"X": synthetic_paw}, ecut=12.0, ecut_rho=96.0)
    assert any("bounded below" in message for message in system.warnings), system.warnings


def test_a_leaking_compensation_charge_is_warned_about(synthetic_paw):
    """The other side of the window: a Gaussian broad enough to spill out of the sphere."""
    config = _config(
        pseudo={"method": "paw", "compensation_shape": "gaussian", "compensation_scale": 2.0 / 9.0}
    )
    system = build_system(_atoms(), config, {"X": synthetic_paw}, ecut=12.0, ecut_rho=96.0)
    assert any("lies outside the" in message for message in system.warnings), system.warnings


def test_the_default_shape_draws_no_complaint(paw_system):
    """A compact shape at full radius leaks nothing and digs a shallower well than the
    Gaussian everyone uses, so neither bound is anywhere near."""
    assert not any(
        "bounded below" in message or "lies outside the" in message
        for message in paw_system.warnings
    ), paw_system.warnings
