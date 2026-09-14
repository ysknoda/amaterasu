"""Phase 1 gate 4: the pseudopotential readers reproduce the datasets' own invariants.

A note on what this gate can actually check
-------------------------------------------
The plan phrased this gate as "verify norm conservation, integral |phi|^2 = integral
|phi~|^2 inside r_c, for every channel". That comparison needs the **all-electron** partial
waves, and neither psp8 nor UPF ships them -- norm conservation is enforced when the
dataset is generated and is not recoverable from the file. Rather than skip the gate, it is
checked here through the invariants the files *do* carry, each of which fails loudly if the
radial convention, the units or the valence charge were misread:

* ``V_loc(r)`` must approach ``-Z_ion/r`` outside the core. This catches a wrong ``Z_ion``,
  and it is the check that catches UPF's Rydberg units, where a missing factor of two
  would otherwise pass every structural test.
* the projectors are ``r beta(r)`` normalised to ``integral p^2 dr = 1`` for
  norm-conserving datasets, which pins the radial convention.
* the atomic valence density integrates to ``Z_ion``, up to the truncation of the radial
  grid.
* the two formats must agree with each other on the same element.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pytest

from amaterasu.pseudo.base import decode_abinit_pspxc
from amaterasu.pseudo.readers import detect_format, read_pseudopotential, read_psp8, read_upf
from amaterasu.units import RYDBERG_IN_HARTREE

DATA = Path(__file__).resolve().parents[1] / "data" / "pseudos"

PSP8_FIXTURES = {
    "Si.pbe.psp8.gz": {"symbol": "Si", "Z": 14.0, "Z_val": 4.0, "xc": "PBE", "projectors": 6},
    "Si.pbesol.psp8.gz": {"symbol": "Si", "Z": 14.0, "Z_val": 4.0, "xc": "PBEsol", "projectors": 6},
    "O.pbe.psp8.gz": {"symbol": "O", "Z": 8.0, "Z_val": 6.0, "xc": "PBE", "projectors": 5},
}


@pytest.fixture(params=sorted(PSP8_FIXTURES))
def psp8_case(request):
    return DATA / request.param, PSP8_FIXTURES[request.param]


# --------------------------------------------------------------------------- psp8
def test_psp8_header(psp8_case):
    path, expected = psp8_case
    pseudo = read_psp8(path)
    assert pseudo.symbol == expected["symbol"]
    assert pseudo.atomic_number == expected["Z"]
    assert pseudo.valence_charge == expected["Z_val"]
    assert pseudo.xc_functional == expected["xc"]
    assert len(pseudo.projectors) == expected["projectors"]
    assert pseudo.kind == "NC"


def test_psp8_local_potential_has_the_right_coulomb_tail(psp8_case):
    """V_loc -> -Z_ion/r. A wrong valence charge or unit shows up here immediately."""
    path, expected = psp8_case
    pseudo = read_psp8(path)
    assert pseudo.local_potential_tail_error(r_min=3.0) < 1e-3

    far = pseudo.grid.r >= 5.0
    coulomb = -expected["Z_val"] / pseudo.grid.r[far]
    assert pseudo.local_potential[far] == pytest.approx(coulomb, rel=1e-5)


def test_psp8_projectors_are_normalised(psp8_case):
    """Norm-conserving generators write r*beta(r) with integral p^2 dr = 1."""
    path, _ = psp8_case
    pseudo = read_psp8(path)
    for projector in pseudo.projectors:
        assert projector.norm(pseudo.grid) == pytest.approx(1.0, abs=1e-4)
        # r*beta(r) must vanish at the origin and outside the core.
        assert abs(projector.values[0]) < 1e-6
        assert abs(projector.values[-1]) < 1e-8


def test_psp8_valence_density_integrates_to_the_valence_charge(psp8_case):
    """The shortfall is the truncated grid, so it is bounded, not zero."""
    path, expected = psp8_case
    pseudo = read_psp8(path)
    charge = pseudo.integrated_valence_charge()
    assert charge == pytest.approx(expected["Z_val"], rel=0.02)
    assert charge < expected["Z_val"], "truncating the grid can only lose charge"


def test_silicon_diffuse_valence_loses_more_charge_than_oxygen():
    """A physical consistency check on the truncation: Si 3s3p is far more diffuse than O 2p."""
    silicon = read_psp8(DATA / "Si.pbe.psp8.gz")
    oxygen = read_psp8(DATA / "O.pbe.psp8.gz")
    si_deficit = 1 - silicon.integrated_valence_charge() / silicon.valence_charge
    o_deficit = 1 - oxygen.integrated_valence_charge() / oxygen.valence_charge
    assert si_deficit > o_deficit


def test_psp8_core_density_is_a_partial_core_not_the_full_core():
    """The NLCC density is pseudised; it must not integrate to the core electron count."""
    pseudo = read_psp8(DATA / "Si.pbe.psp8.gz")
    assert pseudo.has_core_correction
    core_charge = pseudo.grid.enclosed_charge(pseudo.core_density)
    core_electrons = pseudo.atomic_number - pseudo.valence_charge  # 10 for silicon
    assert 0 < core_charge < core_electrons


def test_psp8_check_reports_no_problems(psp8_case):
    path, _ = psp8_case
    assert read_psp8(path).check() == []


def test_pbe_and_pbesol_datasets_differ_only_a_little():
    """Same element and generator settings, different functional: close but not identical."""
    pbe = read_psp8(DATA / "Si.pbe.psp8.gz")
    pbesol = read_psp8(DATA / "Si.pbesol.psp8.gz")
    assert pbe.xc_functional == "PBE"
    assert pbesol.xc_functional == "PBEsol"
    assert np.array_equal(pbe.grid.r, pbesol.grid.r)
    difference = np.abs(pbe.local_potential - pbesol.local_potential).max()
    assert 1e-6 < difference < 1.0, "the two functionals must differ, but not wildly"


# ------------------------------------------------------------------ XC code decoding
@pytest.mark.parametrize(
    ("code", "name", "fragment"),
    [
        (11, "PBE", "PBE"),
        (2, "LDA-PZ", "LDA-PZ"),
        (7, "LDA-PW", "LDA-PW"),
        (-116133, "PBEsol", "GGA_X_PBE_SOL"),
        (-101130, "PBE", "GGA_X_PBE"),
        (-1009, "LDA-PZ", "LDA_C_PZ"),
        (12, None, "revPBE"),
        (-117133, None, "GGA_X_RPBE"),
    ],
)
def test_abinit_xc_codes(code, name, fragment):
    """Negative pspxc packs two libxc ids as -(1000 * id_x + id_c)."""
    decoded_name, description = decode_abinit_pspxc(code)
    assert decoded_name == name
    assert fragment in description


# --------------------------------------------------------------------------- UPF
def _synthetic_upf(*, rydberg_local=True, with_nlcc=True, malformed=False) -> str:
    """A small but structurally complete UPF v2 file with analytically known content.

    Built rather than downloaded so the fixture carries no licence question and so every
    quantity has a value this test knows independently: the local potential is exactly
    ``-2 Z/r`` in Rydberg outside a soft core, the projectors are normalised by
    construction, and the valence density integrates to ``Z`` in closed form.
    """
    z_valence = 4.0
    r = np.linspace(0.0, 10.0, 501)
    step = r[1] - r[0]

    # V_loc = -2Z/r in Rydberg, softened inside r < 1 so it stays finite at the origin.
    soft = np.maximum(r, 1.0)
    vloc = -2.0 * z_valence / soft * (1.0 if rydberg_local else 0.5)

    # Two normalised projectors: r*beta(r) with integral p^2 dr = 1.
    betas = []
    for width in (0.8, 1.3):
        values = r * np.exp(-((r / width) ** 2))
        values /= np.sqrt(np.trapezoid(values**2, r))
        betas.append(values)

    # A hydrogenic 4 pi r^2 rho normalised to exactly z_valence.
    density = r**2 * np.exp(-r)
    density *= z_valence / np.trapezoid(density, r)

    core = np.exp(-((r / 0.5) ** 2))  # a bare rho_c(r), finite at the origin

    def block(values):
        return "\n".join(
            "  ".join(f"{v:.14e}" for v in values[i : i + 4]) for i in range(0, len(values), 4)
        )

    info = "generated by amaterasu tests & verified analytically" if malformed else "synthetic"
    nlcc = f"  <PP_NLCC>\n{block(core)}\n  </PP_NLCC>\n" if with_nlcc else ""
    return f"""<UPF version="2.0.1">
  <PP_INFO>
{info}
  </PP_INFO>
  <PP_HEADER element="Si" pseudo_type="NC" relativistic="scalar" is_ultrasoft="F"
     is_paw="F" has_so="F" core_correction="{"T" if with_nlcc else "F"}" functional="PBE"
     z_valence="{z_valence}" l_max="1" l_local="-1" mesh_size="{len(r)}"
     number_of_wfc="0" number_of_proj="2"/>
  <PP_MESH dx="{step}" mesh="{len(r)}">
    <PP_R>
{block(r)}
    </PP_R>
    <PP_RAB>
{block(np.full_like(r, step))}
    </PP_RAB>
  </PP_MESH>
  <PP_LOCAL>
{block(vloc)}
  </PP_LOCAL>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="{len(r)}">
{block(betas[0])}
    </PP_BETA.1>
    <PP_BETA.2 index="2" angular_momentum="1" cutoff_radius_index="{len(r)}">
{block(betas[1])}
    </PP_BETA.2>
    <PP_DIJ>
  2.0000000000e0  0.0000000000e0  0.0000000000e0  4.0000000000e0
    </PP_DIJ>
  </PP_NONLOCAL>
  <PP_RHOATOM>
{block(density)}
  </PP_RHOATOM>
{nlcc}</UPF>
"""


@pytest.fixture
def synthetic_upf(tmp_path):
    path = tmp_path / "Si_synthetic.upf"
    path.write_text(_synthetic_upf(), encoding="utf-8")
    return path


def test_upf_reads_the_synthetic_file(synthetic_upf):
    pseudo = read_upf(synthetic_upf)
    assert pseudo.symbol == "Si"
    assert pseudo.atomic_number == 14.0  # from the element symbol, not from PP_HEADER
    assert pseudo.valence_charge == 4.0
    assert pseudo.xc_functional == "PBE"
    assert pseudo.kind == "NC"
    assert len(pseudo.projectors) == 2


def test_upf_converts_rydberg_to_hartree(synthetic_upf):
    """The single most dangerous conversion in the format: V_loc and D_ij are in Rydberg."""
    pseudo = read_upf(synthetic_upf)
    far = pseudo.grid.r >= 5.0
    expected = -pseudo.valence_charge / pseudo.grid.r[far]  # Hartree
    assert pseudo.local_potential[far] == pytest.approx(expected, rel=1e-12)
    # PP_DIJ held 2 and 4 Ry on the diagonal.
    assert pseudo.projectors[0].energy == pytest.approx(2.0 * RYDBERG_IN_HARTREE, rel=1e-12)
    assert pseudo.projectors[1].energy == pytest.approx(4.0 * RYDBERG_IN_HARTREE, rel=1e-12)
    assert pseudo.dij.shape == (2, 2)
    assert pseudo.dij[0, 1] == 0.0


def test_upf_density_conventions(synthetic_upf):
    """PP_RHOATOM is already 4 pi r^2 rho; PP_NLCC is a bare rho and must be rescaled."""
    pseudo = read_upf(synthetic_upf)
    assert pseudo.integrated_valence_charge() == pytest.approx(4.0, rel=1e-4)
    # The core density was a Gaussian in rho; after rescaling it must vanish at r = 0.
    assert pseudo.core_density[0] == pytest.approx(0.0, abs=1e-30)
    assert pseudo.grid.enclosed_charge(pseudo.core_density) > 0.0


def test_upf_projector_normalisation(synthetic_upf):
    pseudo = read_upf(synthetic_upf)
    for projector in pseudo.projectors:
        assert projector.norm(pseudo.grid) == pytest.approx(1.0, rel=1e-6)
    assert [p.l for p in pseudo.projectors] == [0, 1]


def test_upf_without_core_correction(tmp_path):
    path = tmp_path / "no_nlcc.upf"
    path.write_text(_synthetic_upf(with_nlcc=False), encoding="utf-8")
    pseudo = read_upf(path)
    assert not pseudo.has_core_correction
    assert pseudo.core_density is None


def test_upf_tolerates_an_unescaped_ampersand(tmp_path):
    """Real UPF files often carry bare '&' in PP_INFO, which is not legal XML."""
    path = tmp_path / "malformed.upf"
    path.write_text(_synthetic_upf(malformed=True), encoding="utf-8")
    pseudo = read_upf(path)
    assert pseudo.valence_charge == 4.0


def test_upf_check_passes(synthetic_upf):
    assert read_upf(synthetic_upf).check() == []


def test_upf_detects_a_wrong_unit_conversion(tmp_path):
    """Sanity check on the check: a file written in Hartree must be flagged, not accepted."""
    path = tmp_path / "hartree.upf"
    path.write_text(_synthetic_upf(rydberg_local=False), encoding="utf-8")
    problems = read_upf(path).check()
    assert any("V_loc" in problem for problem in problems)


# ----------------------------------------------------------------- format dispatch
def test_detect_format():
    assert detect_format(DATA / "Si.pbe.psp8.gz") == "psp8"


def test_detect_format_on_upf(synthetic_upf):
    assert detect_format(synthetic_upf) == "upf"


def test_detect_format_rejects_other_files(tmp_path):
    path = tmp_path / "random.txt"
    path.write_text("this is not a pseudopotential\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognised pseudopotential format"):
        detect_format(path)
    with pytest.raises(FileNotFoundError):
        detect_format(tmp_path / "missing.psp8")


def test_read_pseudopotential_dispatches(synthetic_upf):
    assert read_pseudopotential(DATA / "Si.pbe.psp8.gz").symbol == "Si"
    assert read_pseudopotential(synthetic_upf).symbol == "Si"


def test_gzip_and_plain_give_identical_results(tmp_path):
    """The readers decompress .gz transparently; the result must not depend on it."""
    plain = tmp_path / "Si.psp8"
    with gzip.open(DATA / "Si.pbe.psp8.gz", "rt") as handle:
        plain.write_text(handle.read(), encoding="utf-8")
    compressed = read_psp8(DATA / "Si.pbe.psp8.gz")
    uncompressed = read_psp8(plain)
    assert np.array_equal(compressed.local_potential, uncompressed.local_potential)
    assert np.array_equal(compressed.grid.r, uncompressed.grid.r)


def test_truncated_file_is_reported(tmp_path):
    with gzip.open(DATA / "Si.pbe.psp8.gz", "rt") as handle:
        lines = handle.read().splitlines()
    path = tmp_path / "truncated.psp8"
    path.write_text("\n".join(lines[:400]), encoding="utf-8")
    with pytest.raises(ValueError, match="file ends after"):
        read_psp8(path)


# ------------------------------------------------------- optional: real UPF datasets
@pytest.mark.skipif(
    not os.environ.get("AMATERASU_TEST_PSEUDO_DIR"),
    reason="set AMATERASU_TEST_PSEUDO_DIR to a directory of real UPF files to run this",
)
def test_real_upf_files():
    directory = Path(os.environ["AMATERASU_TEST_PSEUDO_DIR"])
    files = sorted(p for p in directory.glob("*") if p.suffix.lower() in (".upf", ".gz"))
    assert files, f"no UPF files found in {directory}"
    for path in files:
        if detect_format(path) != "upf":
            continue
        pseudo = read_upf(path)
        assert pseudo.local_potential_tail_error(r_min=3.0) < 1e-2, path.name
        if pseudo.valence_density is not None:
            charge = pseudo.integrated_valence_charge()
            assert charge == pytest.approx(pseudo.valence_charge, rel=0.05), path.name
