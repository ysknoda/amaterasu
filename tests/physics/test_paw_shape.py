"""The compensation-charge shape is a free choice, so the total energy must not depend on it.

This is the invariant that a PAW implementation is easiest to get subtly wrong on, because
nothing about a single calculation looks amiss when it fails: the SCF converges, every
multipole is conserved to 1e-8, the biorthogonality holds to 1e-14, and the energy is
simply wrong by a fixed amount that no other test can see. It only shows up by *changing*
the shape and watching the answer move.

What the invariant actually says
--------------------------------
The compensation charge ``Z~^a`` appears twice in the PAW total energy and nowhere else:
once inside the plane-wave Hartree term, on the smooth density ``n~ + Z~``, and once inside
the one-centre term, which subtracts ``1/2 <n~^a + Z~^a|v[n~^a + Z~^a]>`` back on the radial
mesh. It is not in the exchange-correlation term and not in ``v_bar``. So in exact
arithmetic it cancels identically, for any shape carrying the right multipoles.

Two things break the cancellation in practice, and they are worth keeping apart:

*The shape leaves the augmentation sphere.* Then the radial subtraction cannot be complete
-- part of the charge is somewhere the one-centre expansion says nothing about, including
inside a neighbour's sphere -- and no plane-wave cutoff repairs it. This is a formulation
error, and it is what a Gaussian does: 85 meV/atom across the range of widths that are
stable at all, at cutoffs where the Gaussian's own form factor is resolved to 1e-14.

*The grid does not resolve the shape.* Then the two sides are the same integral evaluated
in two bases, one of them truncated. This is ordinary basis incompleteness, it is bounded
by the density cutoff, and it decays as the cutoff rises.

The compactly supported shape removes the first mechanism exactly. What these tests
measure is that neither mechanism explains what is left: the residual does not shrink when
the density cutoff is doubled, by the same two-cutoff argument the Pulay stress is
identified with in ``test_paw_forces.py``, and it is not the leak either, because there is
no leak. It is the dataset's partial waves not spanning the smooth density inside the
sphere. See :mod:`amaterasu.pseudo.paw.shape` for the full comparison, including why the
Gaussian is nonetheless still the default.

Needs the JTH PAW table, which is GPL-3.0 and so not bundled.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import bulk

from amaterasu.config import load_config_mapping
from amaterasu.hamiltonian.system import build_system
from amaterasu.pseudo.readers import read_pseudopotential
from amaterasu.scf.driver import run_scf
from amaterasu.units import HARTREE_IN_EV

pytestmark = [pytest.mark.physics, pytest.mark.slow]

DATASET = Path("~/.amaterasu/pseudos/jth-pbe-v2.0/Si/Si.GGA_PBE-JTH.xml").expanduser()

#: Support radii as fractions of r_c. A 20% range is far more than any real choice between
#: shape functions would span, and every one of them is a legitimate compensation charge.
SCALES = (1.0, 0.9, 0.8)


def _energy(data, shape: str, scale: float, ecut_rho: float) -> float:
    """Total energy per atom of silicon at the equilibrium volume, in eV."""
    atoms = bulk("Si", "diamond", a=5.47)
    config = load_config_mapping(
        {
            "structure": "Si.str",
            "basis": {"ecut": "40 Ry"},
            "kpoints": {"grid": [4, 4, 4]},
            "occupations": {"scheme": "gaussian", "smearing": 0.05},
            "scf": {"max_steps": 120, "tol_energy": 1.0e-9, "tol_density": 1.0e-8},
            "xc": {"functional": "PBE"},
            "pseudo": {
                "method": "paw",
                "compensation_shape": shape,
                "compensation_scale": scale,
            },
        }
    )
    system = build_system(atoms, config, {"Si": data}, ecut=20.0, ecut_rho=ecut_rho)
    result = run_scf(system, config)
    assert result.converged, f"{shape} at scale {scale} did not converge"
    return result.energies.free_energy * HARTREE_IN_EV / len(atoms)


@pytest.fixture(scope="module")
def dataset():
    if not DATASET.exists():
        pytest.skip(
            f"the JTH PAW dataset is not installed at {DATASET}; fetch it with "
            "'amaterasu fetch-pseudos --family jth-pbe-v2.0 --elements Si'"
        )
    return read_pseudopotential(DATASET)


@pytest.fixture(scope="module")
def energies(dataset):
    """Every shape this module compares, run once at identical settings."""
    return {("bessel", scale): _energy(dataset, "bessel", scale, 160.0) for scale in SCALES} | {
        ("gaussian", scale): _energy(dataset, "gaussian", scale, 160.0)
        for scale in (1.0, 6.0 / 9.0)
    }


def test_the_compact_shape_still_moves_the_energy_and_the_cutoff_is_not_why(energies):
    """Measured at ecut = 20 Ha, k = 4^3, over support radii from r_c down to 0.8 r_c:

        support     ecut_rho = 160 Ha       ecut_rho = 320 Ha
        1.00 r_c    -15657.748051, 19 it    -15657.747853, 18 it
        0.90 r_c    -15657.756728, 18 it    -15657.756538, 18 it
        0.80 r_c    -15657.773900, 21 it    -15657.773712, 23 it
        spread          25.85 meV               25.86 meV

    The two columns are the point. Doubling the density cutoff moves the absolute energy by
    0.19 meV/atom and the spread by 0.01, so the residual is not the plane-wave basis
    failing to resolve the compensation charge. It is the term
    ``integral (n~ - n~^a) v[Z~^a]`` inside the sphere -- the dataset's partial waves not
    spanning the smooth density there -- and no cutoff removes it. The bound below is a
    regression guard: reintroducing the leak takes this straight past 85 meV/atom.

    Only the cheaper column is run here; the expensive one is quoted, not repeated.
    """
    values = [energies["bessel", scale] for scale in SCALES]
    spread = 1000.0 * (max(values) - min(values))
    assert spread < 40.0, f"compensation-charge shape moved the energy by {spread:.1f} meV/atom"


def test_the_gaussian_moves_it_further(energies):
    """How much of the shape dependence compact support actually removes.

    A Gaussian's usable widths are ``alpha r_c^2`` between about 6 and 9: broader and it
    spills out of the augmentation sphere, narrower and the energy functional stops being
    bounded below. Across that window it moves silicon's total energy by 85 meV/atom,
    against 26 for the compact shape over a comparable fractional change of width. Both are
    run at the same cutoffs, k-points and thresholds, so the comparison is of the shapes
    alone.

    That factor of three is why ``bessel`` is offered; it is not why it would be the
    default, which the delta factor decides the other way. The module docstring has both
    halves of that.
    """
    gaussian = 1000.0 * abs(energies["gaussian", 1.0] - energies["gaussian", 6.0 / 9.0])
    values = [energies["bessel", scale] for scale in SCALES]
    bessel = 1000.0 * (max(values) - min(values))
    assert gaussian > 2.0 * bessel, (
        f"the compact shape is supposed to be the less shape-dependent one, "
        f"but moved {bessel:.1f} meV/atom against the Gaussian's {gaussian:.1f}"
    )
