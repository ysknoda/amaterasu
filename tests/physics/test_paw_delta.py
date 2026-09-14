"""Phase 4 gate: the delta factor of a PAW calculation against the all-electron reference.

The delta factor (Lejaeghere et al., Science 351, aad3000 (2016)) is the root-mean-square
difference between two equations of state over +-6% of the equilibrium volume, after each
has been shifted to its own minimum. It is insensitive to any constant energy offset,
which is exactly right here: PAW's absolute total energy contains the frozen-core terms
and converges slowly, while the *shape* of E(V) -- the only part physics depends on --
converges much faster.

It is the decisive test for a PAW implementation because it compares against a genuinely
all-electron calculation rather than against another pseudopotential code. A PAW dataset
carries the all-electron partial waves, so getting the one-centre terms right is what
makes this number small; getting them wrong shows up here and almost nowhere else.

The reference values are WIEN2k's, and the target is the value the dataset's own authors
publish for it -- a much sharper statement than the 3 meV/atom the phase gate asks for.

These tests need the JTH PAW table, which is GPL-3.0 and therefore not bundled; they skip
unless it has been fetched. ``amaterasu fetch-pseudos --family jth-pbe-v2.0 --elements Si``
installs it. Oxygen is by far the most expensive of the five -- a four-atom cell, spin
polarised, about twenty minutes a volume against one or two for the cubic elements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from scipy.optimize import least_squares

from amaterasu.config import load_config_mapping
from amaterasu.hamiltonian.system import build_system
from amaterasu.pseudo.readers import read_pseudopotential
from amaterasu.scf.driver import run_scf
from amaterasu.units import BOHR_IN_ANG, HARTREE_IN_EV, HARTREE_PER_BOHR3_IN_GPA

pytestmark = [pytest.mark.physics, pytest.mark.slow]

EV_PER_A3_IN_GPA = HARTREE_PER_BOHR3_IN_GPA / (HARTREE_IN_EV / BOHR_IN_ANG**3)

#: WIEN2k all-electron reference (V0 in A^3/atom, B0 in GPa, B1), transcribed verbatim
#: from ``WIEN2k.txt`` of DeltaCodesDFT (header: "WIEN2k version 13.1, calcDelta package
#: version 3.0"). That file holds **71** element rows; the five used here are quoted
#: exactly as they appear, in order of atomic number:
#:     O    18.5590    51.378   3.89
#:     Al   16.4796    78.077   4.57
#:     Si   20.4530    88.545   4.31
#:     Fe   11.3436   197.652   5.80
#:     Cu   11.9511   141.335   4.86
#: One row in the file (Po) carries a trailing tab; none of these five does. The reference
#: file is LGPL-2.1 and so is not bundled, in keeping with how the PAW tables are handled.
WIEN2K = {
    "O": (18.5590, 51.378, 3.89),
    "Al": (16.4796, 78.077, 4.57),
    "Si": (20.4530, 88.545, 4.31),
    "Fe": (11.3436, 197.652, 5.80),
    "Cu": (11.9511, 141.335, 4.86),
}

#: The elements in the order the results are reported: by atomic number, not alphabetically.
ELEMENTS = ("O", "Al", "Si", "Fe", "Cu")

#: ``ase.build.bulk`` lattice and atoms per conventional cell, for the four cubic elements.
CUBIC = {"Al": ("fcc", 4), "Si": ("diamond", 8), "Fe": ("bcc", 2), "Cu": ("fcc", 4)}

#: Solid oxygen is the one structure ``ase.build.bulk`` cannot make. Cell parameters and
#: fractional coordinates are from ``primCIFs/O.cif`` of DeltaCodesDFT; the four sites are
#: what that file's two sites become under its own symmetry operations. Verified after
#: construction: 4 atoms, C2/m (12), and two O2 molecules -- atoms (0,1) and (2,3), bonded
#: at 1.2318 A in the CIF's own cell and 1.2184 A once scaled to the WIEN2k equilibrium
#: volume, against a gas-phase experimental bond of 1.20752 A (Huber & Herzberg,
#: *Constants of Diatomic Molecules*, 1979). The nearest intermolecular contact is 2.99 A.
OXYGEN_CELLPAR = (4.571478, 4.271628, 4.276819, 90.0, 90.0, 113.278720)
OXYGEN_SITES = np.array(
    [
        [0.35333, 0.43782, 0.00],
        [0.64667, 0.56218, 0.00],
        [0.35333, 0.93782, 0.50],
        [0.64667, 0.06218, 0.50],
    ]
)

#: k-mesh divisions, smearing scheme, smearing width in eV, and the initial moments the
#: protocol asks for -- a single number broadcast over the cell, or one per atom. The
#: DeltaCodesDFT README prescribes ferromagnetic iron and
#: antiferromagnetic oxygen, the antiferromagnetism being *between* the O2 molecules --
#: so the two atoms of each molecule are parallel, which is what makes the molecule the
#: spin triplet it is in the gas phase, and the two molecules oppose each other. That
#: arrangement is not assumed here; ``tests/physics/test_paw_oxygen_spin.py`` measures it
#: against the alternatives.
SETTINGS = {
    "O": (4, "gaussian", 0.05, [1.0, 1.0, -1.0, -1.0]),
    "Al": (12, "marzari-vanderbilt", 0.1, None),
    "Si": (8, "marzari-vanderbilt", 0.1, None),
    "Fe": (12, "marzari-vanderbilt", 0.1, 2.2),
    "Cu": (12, "marzari-vanderbilt", 0.1, None),
}

#: Measured here, at ecut = 20 Ha and ecut_rho = 160 Ha, with the default Gaussian
#: compensation charge. The spread across elements is not the shape function: it tracks how
#: many partial-wave channels each dataset carries. Copper and iron have ``d`` channels and
#: silicon's JTH table does not, and repeating silicon with a ``d``-carrying table -- the
#: pslibrary kjpaw one -- takes its delta factor from 1.868 to 0.326 and its bulk-modulus
#: error from -6.1% to +0.9%. See :mod:`amaterasu.pseudo.paw.shape`.
MEASURED_DELTA = {"O": 0.871, "Al": 0.200, "Si": 1.868, "Fe": 2.699, "Cu": 0.391}

DIRECTORY = Path("~/.amaterasu/pseudos/jth-pbe-v2.0").expanduser()


def structure_at_volume(element: str, volume_per_atom: float) -> Atoms:
    """The Delta-factor structure scaled isotropically to ``volume_per_atom`` in A^3.

    Isotropic because the protocol fixes the cell *shape* and the internal coordinates and
    forbids any relaxation: only the scale varies. For oxygen that means the O2 bond
    stretches with the cell, which is the intended reading of the protocol.
    """
    if element in CUBIC:
        lattice, per_cell = CUBIC[element]
        atoms = bulk(element, lattice, a=(volume_per_atom * per_cell) ** (1.0 / 3.0))
    elif element == "O":
        atoms = Atoms("O4", cell=OXYGEN_CELLPAR, scaled_positions=OXYGEN_SITES, pbc=True)
        scale = (volume_per_atom * len(atoms) / atoms.get_volume()) ** (1.0 / 3.0)
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    else:  # pragma: no cover - defensive
        raise KeyError(f"no Delta-factor structure recorded for {element}")
    moments = SETTINGS[element][3]
    if moments is not None:
        atoms.set_initial_magnetic_moments(np.broadcast_to(moments, len(atoms)))
    return atoms


def birch_murnaghan(volume, e0, v0, b0, b1):
    t = (v0 / volume) ** (2.0 / 3.0) - 1.0
    return e0 + 9.0 * v0 * b0 / 16.0 * (
        t**3 * b1 + t**2 * (6.0 - 4.0 * (v0 / volume) ** (2.0 / 3.0))
    )


def fit(volumes, energies):
    guess = [energies.min(), volumes[int(np.argmin(energies))], 100.0 / EV_PER_A3_IN_GPA, 4.0]
    result = least_squares(
        lambda p: birch_murnaghan(volumes, *p) - energies, guess, xtol=1e-14, ftol=1e-14
    )
    return result.x


def delta_gauge(a, b, npoints=2001):
    """The Lejaeghere delta between two fitted equations of state, in meV/atom."""
    v0 = 0.5 * (a[1] + b[1])
    grid = np.linspace(0.94 * v0, 1.06 * v0, npoints)
    curve_a = birch_murnaghan(grid, 0.0, *a[1:]) - birch_murnaghan(np.array([a[1]]), 0.0, *a[1:])
    curve_b = birch_murnaghan(grid, 0.0, *b[1:]) - birch_murnaghan(np.array([b[1]]), 0.0, *b[1:])
    return 1000.0 * float(
        np.sqrt(np.trapezoid((curve_a - curve_b) ** 2, grid) / (grid[-1] - grid[0]))
    )


@pytest.fixture(scope="module", params=ELEMENTS)
def paw_eos(request):
    """The equation of state of one element, computed with its PAW dataset."""
    element = request.param
    dataset = DIRECTORY / element / f"{element}.GGA_PBE-JTH.xml"
    if not dataset.exists():
        pytest.skip(
            f"the JTH PAW dataset is not installed at {dataset}; fetch it with "
            f"'amaterasu fetch-pseudos --family jth-pbe-v2.0 --elements {element}'"
        )
    data = read_pseudopotential(dataset)
    kgrid, scheme, smearing, moments = SETTINGS[element]
    volumes = WIEN2K[element][0] * np.linspace(0.94, 1.06, 7)
    energies = []
    for volume in volumes:
        atoms = structure_at_volume(element, volume)
        config = load_config_mapping(
            {
                "structure": f"{element}.str",
                "system": {"spin": "none" if moments is None else "polarized"},
                "basis": {"ecut": "40 Ry"},
                "kpoints": {"grid": [kgrid] * 3},
                "occupations": {"scheme": scheme, "smearing": smearing},
                "scf": {"max_steps": 120, "tol_energy": 1.0e-9, "tol_density": 1.0e-7},
                "xc": {"functional": "PBE"},
            }
        )
        system = build_system(atoms, config, {element: data}, ecut=20.0, ecut_rho=160.0)
        result = run_scf(system, config)
        assert result.converged, f"{element}: the SCF did not converge at V = {volume:.3f}"
        energies.append(result.energies.free_energy * HARTREE_IN_EV / len(atoms))
    return element, volumes, np.array(energies)


def test_equation_of_state_matches_the_all_electron_reference(paw_eos):
    """Measured against WIEN2k, with the default Gaussian compensation charge:

        O    V0 = 18.6439 A^3/atom (+0.457%)   B0 =  52.617 GPa (+2.41%)  B1 = 2.43
        Al   V0 = 16.4816 A^3/atom (+0.012%)   B0 =  80.490 GPa (+3.09%)  B1 = 3.62
        Si   V0 = 20.5513 A^3/atom (+0.481%)   B0 =  83.127 GPa (-6.1%)   B1 = 4.61
        Fe   V0 = 11.2795 A^3/atom (-0.565%)   B0 = 188.327 GPa (-4.72%)  B1 = 7.13
        Cu   V0 = 11.9381 A^3/atom (-0.108%)   B0 = 142.453 GPa (+0.79%)  B1 = 4.89

    The volume is what the delta factor mostly rewards and is good throughout. The bulk
    modulus is a second derivative and is harder. Silicon's 6% is the JTH table having no
    ``d`` partial waves, not the compensation charge: the pslibrary table for the same
    element, which does, gives +0.9%.

    ``B1`` is not asserted on. It is a third derivative fitted from seven points over
    +-6%, and oxygen shows what that is worth: 2.43 against WIEN2k's 3.89, while ``V0``
    and ``B0`` -- the two the delta factor is actually sensitive to -- are within 0.5% and
    2.5%. The delta factor itself, 0.871 meV/atom, is the number that matters.
    """
    element, volumes, energies = paw_eos
    mine = fit(volumes, energies)
    v0, b0, _ = WIEN2K[element]
    assert mine[1] == pytest.approx(v0, rel=0.01), f"{element}: equilibrium volume within 1%"
    assert mine[2] * EV_PER_A3_IN_GPA == pytest.approx(b0, rel=0.08), (
        f"{element}: bulk modulus within 8%"
    )


def test_delta_factor(paw_eos):
    """The Phase 4 gate: below 3 meV/atom against the all-electron reference.

    Measured, with the default Gaussian compensation charge and in order of atomic number:
    oxygen 0.871, aluminium 0.200, silicon 1.868, iron 2.699, copper 0.391 meV/atom,
    against VASP 1.9 and GPAW 3.3 for the same quantity. All five clear the gate. The
    per-element bound below is a regression guard set from those measurements, not the
    target.

    The two large ones both come off with a better dataset or a better compensation charge,
    and the pattern says which: silicon's JTH table has no ``d`` partial waves and the
    pslibrary table for the same element does, which takes it to 0.326; iron's *does* have
    them, and switching the compensation charge to the compactly supported ``bessel`` shape
    takes it to 0.825, as it takes copper to 0.064. That is not a rule, though -- the
    pslibrary silicon table also has ``d`` waves and the Gaussian still wins there. See
    :mod:`amaterasu.pseudo.paw.shape`.
    """
    element, volumes, energies = paw_eos
    v0, b0, b1 = WIEN2K[element]
    reference = np.array([0.0, v0, b0 / EV_PER_A3_IN_GPA, b1])
    delta = delta_gauge(fit(volumes, energies), reference)
    assert delta < 3.0, f"{element}: delta = {delta:.3f} meV/atom (VASP 1.9, GPAW 3.3)"
    assert delta < MEASURED_DELTA[element] + 0.3, (
        f"{element}: delta = {delta:.3f} meV/atom, was {MEASURED_DELTA[element]}"
    )
