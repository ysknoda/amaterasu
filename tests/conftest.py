"""Shared fixtures: reference structures with independently known properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.spacegroup import crystal

#: Silicon diamond lattice constant in Angstrom (experiment, extrapolated to 0 K).
SI_LATTICE_CONSTANT = 5.4307

#: Properties of the Si primitive cell, each derivable in closed form or from experiment.
SI_REFERENCE = {
    "nearest_neighbour": np.sqrt(3) / 4 * SI_LATTICE_CONSTANT,  # 2.35156 A (exp 2.352)
    "volume": SI_LATTICE_CONSTANT**3 / 4,  # 40.0412 A^3
    "density": 2.3294,  # g/cm^3 (exp 2.329)
    "spacegroup": 227,  # Fd-3m
    "n_operations": 48,
}

SI_STRUCTURE_TEXT = """\
# Silicon, diamond structure, primitive FCC cell
units       angstrom
scale       1.0

%block lattice_vectors
  0.00000   2.71535   2.71535
  2.71535   0.00000   2.71535
  2.71535   2.71535   0.00000
%endblock lattice_vectors

%block atomic_positions fractional
   Si    0.0000   0.0000   0.0000
   Si    0.2500   0.2500   0.2500
%endblock atomic_positions
"""


@pytest.fixture
def si_text() -> str:
    """The amaterasu-format text for the Si primitive cell."""
    return SI_STRUCTURE_TEXT


@pytest.fixture
def si_file(tmp_path, si_text):
    """The Si structure written to a temporary file."""
    path = tmp_path / "Si.str"
    path.write_text(si_text, encoding="utf-8")
    return path


@pytest.fixture
def si_atoms():
    """Si diamond as an ase.Atoms, built independently of amaterasu's own reader."""
    return bulk("Si", "diamond", a=SI_LATTICE_CONSTANT)


@pytest.fixture
def nacl_atoms():
    """NaCl rocksalt, space group Fm-3m (225)."""
    return bulk("NaCl", "rocksalt", a=5.64)


@pytest.fixture
def rutile_atoms():
    """Rutile TiO2, space group P4_2/mnm (136); shortest Ti-O is 1.9486 A (exp 1.949)."""
    return crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (0.30478, 0.30478, 0)],
        spacegroup=136,
        cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90],
    )


#: The bundled test pseudopotentials, so no test in the suite needs a network connection.
PSEUDO_DATA = Path(__file__).resolve().parent / "data" / "pseudos"


@pytest.fixture
def si_config_text() -> str:
    """A minimal but complete calculation input for Si, using the bundled datasets."""
    return f"""\
structure: Si.str
solver:    scf
xc:        {{functional: PBE}}
pseudo:
  method:    nc
  directory: {PSEUDO_DATA}
  files:     {{Si: Si.pbe.psp8.gz}}
basis:     {{ecut: 60}}
kpoints:   {{scheme: monkhorst-pack, grid: [8, 8, 8]}}
"""


@pytest.fixture(scope="session")
def si_pseudo():
    """The bundled PseudoDojo Si PBE dataset, read once per session."""
    from amaterasu.pseudo.readers import read_pseudopotential

    return read_pseudopotential(PSEUDO_DATA / "Si.pbe.psp8.gz")


@pytest.fixture(scope="session")
def o_pseudo():
    """The bundled PseudoDojo O PBE dataset, read once per session."""
    from amaterasu.pseudo.readers import read_pseudopotential

    return read_pseudopotential(PSEUDO_DATA / "O.pbe.psp8.gz")


@pytest.fixture
def si_scf_config():
    """A small, fast Si input: enough basis and k-points to be physical, not to be accurate."""
    from amaterasu.config.schema import Config

    config = Config(structure="-", solver="scf")
    config.xc.functional = "PBE"
    config.kpoints.grid = [2, 2, 2]
    config.occupations.scheme = "fixed"
    config.scf.max_steps = 40
    config.scf.tol_energy = 1e-8
    config.scf.tol_density = 1e-7
    return config


# ------------------------------------------------------------------ PAW test fixtures
#
# The PAW unit tests run against a *synthetic* dataset built here rather than a published
# one. Two reasons: the published PAW tables are GPL-3.0 (JTH) or GPL (GPAW) and bundling
# one would propagate that licence into an Apache-2.0 repository, and a dataset whose
# partial waves are written down in closed form lets a test assert what the answer must
# be instead of only that nothing changed. The physics tests use the real tables and skip
# when they are not installed.

#: Augmentation radius of the synthetic dataset, in Bohr.
SYNTHETIC_RCUT = 1.8


def _synthetic_pawxml_text() -> str:
    """A valid PAW-XML v0.7 document with analytically constructed partial waves.

    The construction satisfies every invariant a real dataset does: the partial waves
    match outside ``r_c`` (the smooth ones differ by a bump that vanishes to third order
    there), the projectors are biorthogonal to the smooth waves by explicit inversion of
    their overlap, the core density integrates to the declared number of core electrons,
    and the smooth core joins it smoothly at ``r_c``.
    """
    import numpy as np

    a, d, n = 2.0e-3, 0.025, 400
    index = np.arange(n, dtype=float)
    r = a * (np.exp(d * index) - 1.0)
    dr_di = a * d * np.exp(d * index)
    rc = SYNTHETIC_RCUT
    inside = r < rc
    bump = np.where(inside, (1.0 - (r / rc) ** 2) ** 4, 0.0)
    localised = np.where(inside, (1.0 - (r / rc) ** 2) ** 3, 0.0)

    channels = [(0, 0.60, 0.35), (0, 1.10, -0.20), (1, 0.80, 0.25)]
    ae = np.array([r**l * np.exp(-beta * r**2) for l, beta, _ in channels])
    ps = np.array(
        [
            wave + amplitude * r**l * bump
            for wave, (l, _, amplitude) in zip(ae, channels, strict=True)
        ]
    )

    from scipy.integrate import simpson

    def integrate(values):
        return simpson(values * dr_di, dx=1.0)

    # Projectors: localised trial functions, biorthogonalised within each l block. Two
    # channels of the same l need two *independent* trials, so successive ones carry an
    # extra factor of (r/r_c)^2.
    trial = np.zeros_like(ae)
    seen: dict[int, int] = {}
    for i, (l, _, _) in enumerate(channels):
        order = seen.get(l, 0)
        seen[l] = order + 1
        trial[i] = r**l * localised * (r / rc) ** (2 * order)
    projectors = np.zeros_like(trial)
    # Biorthogonalise with the same truncated quadrature the reader checks with, so that
    # the file satisfies <p_i|phit_j> = delta_ij to machine precision rather than to the
    # difference between two Simpson panels straddling r_c.
    stop = min(int(np.searchsorted(r, rc, side="right")) + 1, n)

    def inside_sphere(values):
        return simpson(values[:stop] * dr_di[:stop], dx=1.0)

    for l in sorted(seen):
        rows = [i for i, c in enumerate(channels) if c[0] == l]
        overlap = np.array([[inside_sphere(trial[i] * ps[j] * r**2) for j in rows] for i in rows])
        # <p_a|phit_c> = sum_b M_ab <q_b|phit_c> = (M S)_ac = delta_ac, so M = S^-1.
        coefficients = np.linalg.inv(overlap)
        for a_index, i in enumerate(rows):
            projectors[i] = sum(coefficients[a_index, b] * trial[rows[b]] for b in range(len(rows)))

    # Core densities, stored as the l = 0 expansion coefficient (times Y_00 for the real one).
    core = np.exp(-8.0 * r**2)
    core *= 2.0 / (np.sqrt(4.0 * np.pi) * integrate(core * r**2))
    # Smooth core: a + b r^2 + c r^4 inside r_c, matching value and two derivatives at r_c.
    x = rc**2
    value, slope, curve = np.exp(-8.0 * x), -8.0 * np.exp(-8.0 * x), 64.0 * np.exp(-8.0 * x)
    matrix = np.array([[1.0, x, x**2], [0.0, 1.0, 2.0 * x], [0.0, 0.0, 2.0]])
    coefficients = np.linalg.solve(matrix, [value, slope, curve])
    scale = 2.0 / (np.sqrt(4.0 * np.pi) * integrate(np.exp(-8.0 * r**2) * r**2))
    smooth_core = np.where(
        inside,
        scale * (coefficients[0] + coefficients[1] * r**2 + coefficients[2] * r**4),
        core,
    )

    valence = np.exp(-0.9 * r**2)
    valence *= 4.0 / (np.sqrt(4.0 * np.pi) * integrate(valence * r**2))
    vbar = -0.6 * localised

    # Kinetic-energy differences, from the same expression the tests use.
    def kinetic(stack):
        slope = np.gradient(stack, axis=-1) / dr_di
        size = len(stack)
        out = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if channels[i][0] != channels[j][0]:
                    continue
                l = channels[i][0]
                out[i, j] = 0.5 * integrate(
                    r**2 * slope[i] * slope[j] + l * (l + 1) * stack[i] * stack[j]
                )
        return out

    differences = kinetic(ae) - kinetic(ps)

    def block(values):
        return "\n".join(
            "  ".join(f"{v:.14e}" for v in values[k : k + 4]) for k in range(0, len(values), 4)
        )

    states, waves = [], []
    for i, (l, _, _) in enumerate(channels):
        identifier = f"X{i + 1}"
        occupation = [2.0, 0.0, 2.0][i]
        energy = [-0.5, 0.8, -0.2][i]
        principal = f' n="{2 + l}"' if occupation > 0 else ""
        states.append(
            f'    <state{principal} l="{l}" f="{occupation}" rc="{rc}" e="{energy}" id="{identifier}"/>'
        )
        for tag, values in (
            ("ae_partial_wave", ae[i]),
            ("pseudo_partial_wave", ps[i]),
            ("projector_function", projectors[i]),
        ):
            waves.append(f'  <{tag} state="{identifier}" grid="log1">\n{block(values)}\n  </{tag}>')

    parts = [
        '<paw_dataset version="0.7">',
        '  <pw_ecut low="12.00" medium="16.00" high="20.00"/>',
        '  <atom symbol="X" Z="6.00" core="2.00" valence="4.00"/>',
        '  <xc_functional type="LDA" name="PW"/>',
        '  <generator type="non-relativistic" name="amaterasu-test"/>',
        f'  <paw_radius rc="{rc}"/>',
        "  <valence_states>",
        *states,
        "  </valence_states>",
        f'  <radial_grid eq="r=a*(exp(d*i)-1)" a="{a:.14e}" d="{d:.14e}" istart="0" '
        f'iend="{n - 1}" id="log1"/>',
        f'  <shape_function type="gauss" rc="{rc}"/>',
        f'  <ae_core_density grid="log1">\n{block(core)}\n  </ae_core_density>',
        f'  <pseudo_core_density grid="log1">\n{block(smooth_core)}\n  </pseudo_core_density>',
        f'  <pseudo_valence_density grid="log1">\n{block(valence)}\n  </pseudo_valence_density>',
        f'  <zero_potential grid="log1">\n{block(vbar)}\n  </zero_potential>',
        *waves,
        f"  <kinetic_energy_differences>\n{block(differences.ravel())}\n"
        "  </kinetic_energy_differences>",
        "</paw_dataset>",
    ]
    return "\n".join(parts) + "\n"


@pytest.fixture(scope="session")
def synthetic_pawxml_text() -> str:
    return _synthetic_pawxml_text()


@pytest.fixture(scope="session")
def synthetic_pawxml_file(tmp_path_factory, synthetic_pawxml_text) -> Path:
    path = tmp_path_factory.mktemp("paw") / "X.synthetic.xml"
    path.write_text(synthetic_pawxml_text, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def synthetic_paw(synthetic_pawxml_file):
    """The synthetic dataset, read back through the real PAW-XML reader."""
    from amaterasu.pseudo.readers.pawxml import read_pawxml

    return read_pawxml(synthetic_pawxml_file)


# ------------------------------------------------------- synthetic UPF-PAW (Kresse-Joubert)
#
# The same synthetic dataset again, written in the *other* PAW formulation. A pslibrary
# kjpaw file does not carry v_bar; it carries the local ionic pseudopotential instead, and
# the reader has to invert the transformation between them. Emitting the transformation
# forwards here and letting the reader run it backwards turns that into a round trip, which
# is what makes the conversion testable without a GPL-licensed file to compare against.


def _synthetic_upf_paw_text(data) -> str:
    """Re-emit a PAW dataset as a Kresse-Joubert UPF v2 file.

    ``data`` is the synthetic dataset as the PAW-XML reader returns it. The element is
    written as carbon because a UPF header carries no nuclear charge and the reader has to
    look it up from the symbol; the synthetic dataset's Z = 6, 2 core and 4 valence
    electrons are carbon's anyway.
    """
    import numpy as np

    from amaterasu.pseudo.paw.kresse_joubert import (
        ROOT_FOUR_PI,
        radial_xc_potential,
    )
    from amaterasu.pseudo.paw.radial import integrate, radial_poisson
    from amaterasu.pseudo.paw.shape import GaussianShape
    from amaterasu.units import RYDBERG_IN_HARTREE
    from amaterasu.xc.driver import XCFunctional

    grid = data.grid
    r = grid.r
    # A UPF file locates the augmentation sphere by mesh index, so the radius the reader
    # will see is the last grid point inside r_c rather than r_c itself. Build the file
    # around that radius so the round trip is exact rather than nearly so.
    sphere_index = int(np.searchsorted(r, data.rcut, side="right")) - 1
    shape = GaussianShape.for_radius(float(r[sphere_index]))
    xc = XCFunctional(data.xc_functional or "LDA-PW")

    smooth_core = integrate(grid, ROOT_FOUR_PI * data.pseudo_core_density * r**2)
    smooth_valence = integrate(grid, ROOT_FOUR_PI * data.pseudo_valence_density * r**2)
    core_charge = data.core_electrons - data.atomic_number - smooth_core
    valence_charge = data.valence_charge - smooth_valence - smooth_core

    def smeared(charge):
        return charge * shape.radial(0, r) / ROOT_FOUR_PI

    smooth = data.pseudo_valence_density + data.pseudo_core_density
    # The transformation of the PAW-XML specification, run forwards.
    local = (
        radial_poisson(grid, 0, data.pseudo_core_density + smeared(core_charge))
        + data.zero_potential
        + radial_xc_potential(xc, grid, smooth)
        - radial_xc_potential(xc, grid, smooth + smeared(valence_charge))
    )

    def block(values):
        rows = ["  ".join(f"{v:.14e}" for v in values[k : k + 4]) for k in range(0, len(values), 4)]
        return "\n".join(rows)

    count = data.n_waves
    parts = [
        '<UPF version="2.0.1">',
        "  <PP_INFO>synthetic Kresse-Joubert dataset for the amaterasu test suite</PP_INFO>",
        '  <PP_HEADER generated="amaterasu-test" element="C" pseudo_type="PAW" '
        'relativistic="scalar" is_ultrasoft="true" is_paw="true" core_correction="true" '
        'has_so="false" functional=" SLA  PW" '
        f'z_valence="{data.valence_charge:.12e}" l_max="{data.l_max}" '
        f'l_max_rho="{2 * data.l_max}" mesh_size="{len(r)}" '
        f'number_of_wfc="0" number_of_proj="{count}" '
        'wfc_cutoff="2.4e1" rho_cutoff="9.6e1"/>',
        "  <PP_MESH>",
        f"    <PP_R>\n{block(r)}\n    </PP_R>",
        f"    <PP_RAB>\n{block(grid.dr_di)}\n    </PP_RAB>",
        "  </PP_MESH>",
        f"  <PP_NLCC>\n{block(data.pseudo_core_density / ROOT_FOUR_PI)}\n  </PP_NLCC>",
        f"  <PP_LOCAL>\n{block(local / ROOT_FOUR_PI / RYDBERG_IN_HARTREE)}\n  </PP_LOCAL>",
        "  <PP_NONLOCAL>",
    ]
    for i, wave in enumerate(data.waves):
        parts.append(
            f'    <PP_BETA.{i + 1} index="{i + 1}" label="{wave.label}" '
            f'angular_momentum="{wave.l}" cutoff_radius_index="{sphere_index}">'
            f"\n{block(wave.projector * r)}\n    </PP_BETA.{i + 1}>"
        )
    identity = np.zeros((count, count))
    parts += [
        f"    <PP_DIJ>\n{block(identity.ravel())}\n    </PP_DIJ>",
        f'    <PP_AUGMENTATION q_with_l="true" nqf="0" shape="PSQ" '
        f'cutoff_r_index="{sphere_index}" l_max_aug="{2 * data.l_max}">',
        "    </PP_AUGMENTATION>",
        "  </PP_NONLOCAL>",
        "  <PP_FULL_WFC>",
    ]
    for i, wave in enumerate(data.waves):
        parts.append(
            f'    <PP_AEWFC.{i + 1} index="{i + 1}" label="{wave.label}" l="{wave.l}">'
            f"\n{block(wave.ae * r)}\n    </PP_AEWFC.{i + 1}>"
        )
    for i, wave in enumerate(data.waves):
        parts.append(
            f'    <PP_PSWFC.{i + 1} index="{i + 1}" label="{wave.label}" l="{wave.l}">'
            f"\n{block(wave.ps * r)}\n    </PP_PSWFC.{i + 1}>"
        )
    occupations = np.array([w.occupation for w in data.waves])
    parts += [
        "  </PP_FULL_WFC>",
        f"  <PP_RHOATOM>\n{block(data.pseudo_valence_density * ROOT_FOUR_PI * r**2)}\n"
        "  </PP_RHOATOM>",
        '  <PP_PAW paw_data_format="2" core_energy="-1.0e1">',
        f"    <PP_OCCUPATIONS>\n{block(occupations)}\n    </PP_OCCUPATIONS>",
        f"    <PP_AE_NLCC>\n{block(data.ae_core_density / ROOT_FOUR_PI)}\n    </PP_AE_NLCC>",
        f"    <PP_AE_VLOC>\n{block(np.zeros_like(r))}\n    </PP_AE_VLOC>",
        "  </PP_PAW>",
        "</UPF>",
    ]
    return "\n".join(parts) + "\n"


@pytest.fixture(scope="session")
def synthetic_upf_paw_file(tmp_path_factory, synthetic_paw) -> Path:
    path = tmp_path_factory.mktemp("upfpaw") / "C.synthetic-kjpaw.UPF"
    path.write_text(_synthetic_upf_paw_text(synthetic_paw), encoding="utf-8")
    return path
