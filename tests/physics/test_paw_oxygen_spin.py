"""Oxygen's magnetic state, which the delta-factor protocol prescribes but does not prove.

The DeltaCodesDFT README asks for solid oxygen to be run antiferromagnetically, "between
the O2 molecules". That phrase carries a physical claim worth checking rather than
assuming, because it is the whole reason the four-atom cell is run spin-polarised at all:
each O2 molecule stays a **spin triplet**, and it is only the relative orientation of the
two molecules that alternates.

The triplet is not an arbitrary starting guess. O2 has twelve valence electrons filling
``sigma_2s^2 sigma*_2s^2 sigma_2p^2 pi_2p^4 pi*_2p^2``, and the last two go one each into
the two degenerate ``pi*`` orbitals with parallel spins, giving the ``X ^3 Sigma_g^-``
ground state with a total moment of exactly 2 mu_B. That integer is what makes the claim
testable: a calculation that reproduces it has the orbital filling right, and one that
does not has fallen into a different state whatever its energy says.

Two things are therefore measured here, on a PAW dataset, at the settings the delta-factor
run uses:

* one molecule in a box -- the total moment must be 2 mu_B, and the triplet must lie below
  the spin-restricted solution;
* the crystal -- of the four ways to arrange the moments, the protocol's must be the one
  the functional actually prefers, and the ferromagnetic arrangement must carry 4 mu_B,
  two per molecule, since there the symmetry does not cancel the total.

The per-sphere moments are worth a word. A PAW sphere for oxygen has ``r_c = 1.415`` Bohr,
and the 2p spin density reaches well outside it, so the moment inside a sphere is only
about a fifth of the atom's. The molecular run calibrates exactly that: a total moment of
2 mu_B, an unambiguous 1 mu_B per atom, shows up as 0.223 inside the spheres. The crystal's
0.212 is therefore a full atomic moment too, not a collapsed one.

Needs the JTH PAW table, which is GPL-3.0 and so not bundled; skips without it. These are
slow: five self-consistent calculations, a few tens of minutes each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from amaterasu.config import load_config_mapping
from amaterasu.hamiltonian.system import build_system
from amaterasu.pseudo.readers import read_pseudopotential
from amaterasu.scf.driver import compute_occupancies, run_scf
from amaterasu.units import HARTREE_IN_EV

from .test_paw_delta import WIEN2K, structure_at_volume

pytestmark = [pytest.mark.physics, pytest.mark.slow]

DATASET = Path("~/.amaterasu/pseudos/jth-pbe-v2.0/O/O.GGA_PBE-JTH.xml").expanduser()

#: Experimental ``a ^1 Delta_g <- X ^3 Sigma_g^-`` excitation in eV (Huber & Herzberg,
#: *Constants of Diatomic Molecules*, 1979). A spin-restricted single determinant is not
#: that state, so this fixes the magnitude to expect, not the value.
SINGLET_TRIPLET_EXPERIMENT = 0.98

#: Gas-phase O2 bond length in Angstrom (Huber & Herzberg). The molecular run below uses
#: the *crystal's* bond instead, so that it speaks directly to the crystal calculation.
BOND_EXPERIMENT = 1.20752

#: Cube edge in Angstrom for the isolated molecule: 1.22 A of bond and 6.8 A of vacuum.
BOX = 8.0


def _skip_without_dataset():
    if not DATASET.exists():
        pytest.skip(
            f"the JTH PAW dataset is not installed at {DATASET}; fetch it with "
            "'amaterasu fetch-pseudos --family jth-pbe-v2.0 --elements O'"
        )


def _run(atoms, spin, kgrid):
    """One self-consistent calculation at the delta-factor settings."""
    config = load_config_mapping(
        {
            "structure": "O.str",
            "system": {"spin": spin},
            "basis": {"ecut": "40 Ry"},
            "kpoints": {"grid": [kgrid] * 3},
            "occupations": {"scheme": "gaussian", "smearing": 0.05},
            "scf": {"max_steps": 200, "tol_energy": 1.0e-9, "tol_density": 1.0e-7},
            "xc": {"functional": "PBE"},
        }
    )
    data = read_pseudopotential(DATASET)
    system = build_system(atoms, config, {"O": data}, ecut=20.0, ecut_rho=160.0)
    result = run_scf(system, config)
    assert result.converged, "the SCF did not converge"
    return system, result, result.energies.free_energy * HARTREE_IN_EV / len(atoms)


def sphere_moments(system, result):
    """Moment held inside each PAW sphere, from the occupancy matrices.

    An antiferromagnet's *total* moment is zero, and so is a non-magnetic solution's, so
    the converged total says nothing about which one was found. This does.
    """
    occ = compute_occupancies(system, result.wavefunctions, result.occupations)
    overlap = system.paw.overlap
    return np.array(
        [
            float(np.einsum("ij,ij->", occ[0, b, b], overlap[b, b]))
            - float(np.einsum("ij,ij->", occ[1, b, b], overlap[b, b]))
            for b in (slice(start, start + size) for start, size in system.paw.blocks)
        ]
    )


def test_the_cell_holds_two_o2_molecules():
    """The protocol's antiferromagnetism is *between* molecules, so they must be identified.

    Cheap, and it guards the thing the rest of the file assumes: that atoms (0,1) and
    (2,3) are the two molecules, which is what makes ``[+1 +1 -1 -1]`` mean "triplets,
    antialigned" rather than something else entirely.
    """
    atoms = structure_at_volume("O", WIEN2K["O"][0])
    assert len(atoms) == 4
    distances = atoms.get_all_distances(mic=True)
    bonds = [(0, 1), (2, 3)]
    for i, j in bonds:
        assert distances[i, j] == pytest.approx(1.2184, abs=1e-3)
        # Each atom's nearest neighbour must be its own partner, or the pairing is wrong.
        others = np.where(np.arange(4) == i, np.inf, distances[i])
        assert int(np.argmin(others)) == j
    # The molecules have to be well separated, or "between the molecules" is meaningless.
    assert distances[0, 2] == pytest.approx(2.9894, abs=1e-3)
    assert np.allclose(atoms.get_initial_magnetic_moments(), [1.0, 1.0, -1.0, -1.0])
    # The crystal's bond is within 1% of the gas-phase measurement.
    assert distances[0, 1] == pytest.approx(BOND_EXPERIMENT, rel=0.01)


@pytest.fixture(scope="module")
def molecule():
    """One O2 in a box, spin-polarised and spin-restricted."""
    _skip_without_dataset()
    bond = structure_at_volume("O", WIEN2K["O"][0]).get_distance(0, 1, mic=True)
    base = Atoms(
        "O2",
        positions=[[0.0, 0.0, -bond / 2], [0.0, 0.0, bond / 2]],
        cell=[BOX] * 3,
        pbc=True,
    )
    base.center()
    polarised = base.copy()
    polarised.set_initial_magnetic_moments([1.0, 1.0])
    return _run(polarised, "polarized", 1), _run(base, "none", 1)


def test_the_isolated_molecule_is_a_triplet(molecule):
    """Total moment 2.0000 mu_B, measured -- the integer that identifies ``X ^3 Sigma_g^-``.

    Also the calibration for the crystal: 1 mu_B per atom shows up as 0.223 inside the
    PAW spheres, because oxygen's ``r_c`` of 1.415 Bohr holds only part of the 2p density.
    """
    (system, result, _), _ = molecule
    assert result.magnetization == pytest.approx(2.0, abs=1e-4)
    moments = sphere_moments(system, result)
    assert moments == pytest.approx([0.2229, 0.2229], abs=5e-3)


def test_the_triplet_is_the_stable_state(molecule):
    """Measured: the spin-restricted solution lies 1.087 eV per molecule above the triplet.

    The nearest observable is the ``a ^1 Delta_g`` excitation at 0.98 eV; a restricted
    single determinant is not that state, so the test is on the magnitude -- the triplet
    must win, and win by about an electronvolt rather than by a few meV.
    """
    (_, _, triplet), (_, _, restricted) = molecule
    split = (restricted - triplet) * 2.0  # per molecule
    assert split > 0.0, "the spin-polarised triplet must be the lower state"
    assert split == pytest.approx(SINGLET_TRIPLET_EXPERIMENT, abs=0.3)


@pytest.fixture(scope="module")
def crystal():
    """The four spin arrangements at the WIEN2k equilibrium volume."""
    _skip_without_dataset()
    base = structure_at_volume("O", WIEN2K["O"][0])
    arrangements = {
        "afm": [1.0, 1.0, -1.0, -1.0],  # triplet molecules, antialigned -- the protocol
        "fm": [1.0, 1.0, 1.0, 1.0],  # triplet molecules, aligned
        "intra-afm": [1.0, -1.0, 1.0, -1.0],  # moments opposed *within* a molecule
        "none": None,  # spin-restricted
    }
    out = {}
    for label, moments in arrangements.items():
        atoms = base.copy()
        if moments is not None:
            atoms.set_initial_magnetic_moments(moments)
        out[label] = _run(atoms, "none" if moments is None else "polarized", 4)
    return out


def test_the_crystal_prefers_the_protocols_arrangement(crystal):
    """Measured at V0, in meV/atom above the lowest:

        afm         0        (the protocol: triplet molecules, antialigned)
        fm        +55.77     (triplet molecules, aligned)
        intra-afm +305.38    (moments opposed within a molecule)
        none      +305.38    (spin-restricted)

    The protocol prescribes antiferromagnetism between the molecules; this is the check
    that the functional agrees, rather than the arrangement being imposed on it. The
    ordering is the physical one: flipping a whole molecule costs an intermolecular
    exchange energy, 56 meV/atom, while breaking the triplet costs the whole magnetic
    stabilisation, 305 meV/atom -- five and a half times as much.
    """
    energies = {label: value[2] for label, value in crystal.items()}
    best = min(energies, key=energies.get)
    assert best == "afm", f"the protocol asks for afm but {best} came out lower: {energies}"
    # Breaking the triplet costs far more than flipping a whole molecule.
    assert energies["intra-afm"] - energies["afm"] > 4.0 * (energies["fm"] - energies["afm"])
    # Magnetism is worth having at all: the restricted solution is far above.
    assert energies["none"] - energies["afm"] == pytest.approx(0.30538, abs=0.02)


def test_opposing_the_moments_inside_a_molecule_destroys_them(crystal):
    """Starting from ``[+1 -1 +1 -1]`` does not give an intramolecular antiferromagnet.

    It gives the non-magnetic solution -- the same energy to every digit, and per-sphere
    moments of order 1e-8. There is no self-consistent state with the two atoms of an O2
    opposed, which is the sharpest form of the statement that the molecule is a triplet:
    the moment is not a property of the atoms that happens to align, it is the molecule's.
    """
    (_, broken_result, broken), (_, _, restricted) = crystal["intra-afm"], crystal["none"]
    assert broken == pytest.approx(restricted, abs=1e-5)
    system = crystal["intra-afm"][0]
    assert np.abs(sphere_moments(system, broken_result)).max() < 1e-6


def test_the_ferromagnetic_arrangement_carries_two_bohr_magnetons_per_molecule(crystal):
    """Four mu_B in a four-atom cell: two triplet molecules, aligned.

    This is the decisive measurement for the triplet claim in the *crystal*, because the
    antiferromagnetic total is zero by symmetry and says nothing. Measured: +4.0000 mu_B,
    with per-sphere moments of +0.2229 -- the same number the isolated triplet molecule
    gives, which is the cross-check that the crystal's molecules are the same objects. The
    antiferromagnet then shows 0.2123 with alternating sign, a full moment slightly reduced
    by the intermolecular overlap rather than a collapsed one.
    """
    _, result, _ = crystal["fm"]
    assert result.magnetization == pytest.approx(4.0, abs=1e-3)
    afm_system, afm_result, _ = crystal["afm"]
    moments = sphere_moments(afm_system, afm_result)
    assert moments == pytest.approx([0.2123, 0.2123, -0.2123, -0.2123], abs=5e-3)
    # Zero by symmetry. The bound is the density convergence threshold, not machine
    # precision: the run logged +0.0000 at four decimals, so below 5e-5, and tol_density
    # is 1e-7. Anything this test needs to exclude -- a partial or collapsed moment --
    # is three orders of magnitude larger.
    assert afm_result.magnetization == pytest.approx(0.0, abs=1e-3)
