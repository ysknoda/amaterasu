"""Resolving pseudopotentials for a calculation, and refusing inconsistent combinations.

These tests do not touch the network: file resolution runs against the local fixtures and
manifest handling against a cached manifest written into ``tmp_path``. The one test that
would download is marked ``physics`` so it stays out of the fast suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amaterasu.config import load_config_mapping
from amaterasu.pseudo.registry import (
    KNOWN_FAMILIES,
    PseudoResolutionError,
    default_family,
    load_manifest,
    load_pseudopotentials,
    recommended_cutoff,
    resolve_cutoffs,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "pseudos"


def make_config(**overrides):
    """A minimal config pointing at the bundled test datasets."""
    pseudo = {"method": "nc", "directory": str(DATA), "files": {"Si": "Si.pbe.psp8.gz"}}
    pseudo.update(overrides.pop("pseudo", {}))
    body = {
        "structure": "Si.str",
        "basis": {"ecut": 60},
        "xc": {"functional": "PBE"},
        "pseudo": pseudo,
    }
    body.update(overrides)
    return load_config_mapping(body)


# --------------------------------------------------------------------- family table
def test_known_families_are_self_consistent():
    for name, family in KNOWN_FAMILIES.items():
        assert family.name == name
        assert family.kind in ("nc", "paw")
        assert family.xc in ("PBE", "PBEsol", "LDA-PW", "LDA-PZ")
        assert family.manifest_url.startswith("https://")
        assert family.citation


def test_default_family_matches_the_functional():
    assert default_family("PBE", "nc") == "pseudodojo-nc-sr-pbe-standard-v0.4"
    assert default_family("PBEsol", "nc") == "pseudodojo-nc-sr-pbesol-standard-v0.4"
    # PBEsol PAW exists only as pslibrary UPF, which is not a fetchable family.
    assert default_family("PBEsol", "paw") is None


def test_file_url_handles_semicore_basenames():
    """Fe's recommended dataset is Fe-sp.psp8, which lives in the Fe/ directory."""
    family = KNOWN_FAMILIES["pseudodojo-nc-sr-pbe-standard-v0.4"]
    assert family.file_url("Fe-sp.psp8").endswith("/Fe/Fe-sp.psp8")
    assert family.file_url("Si.psp8").endswith("/Si/Si.psp8")


# ------------------------------------------------------------------- explicit files
def test_explicit_files_are_loaded():
    resolved = load_pseudopotentials(make_config(), ["Si"], download=False)
    assert resolved.pseudopotentials["Si"].symbol == "Si"
    assert resolved.total_valence_electrons == 4.0


def test_multiple_species():
    config = make_config(pseudo={"files": {"Si": "Si.pbe.psp8.gz", "O": "O.pbe.psp8.gz"}})
    resolved = load_pseudopotentials(config, ["Si", "O", "Si"], download=False)
    assert sorted(resolved.pseudopotentials) == ["O", "Si"]
    assert resolved.total_valence_electrons == 10.0


def test_missing_file_is_reported():
    config = make_config(pseudo={"files": {"Si": "nope.psp8"}})
    with pytest.raises(PseudoResolutionError, match="no such file"):
        load_pseudopotentials(config, ["Si"], download=False)


def test_wrong_element_is_reported():
    """A dataset for the wrong element parses fine, so it has to be caught explicitly."""
    config = make_config(pseudo={"files": {"Si": "O.pbe.psp8.gz"}})
    with pytest.raises(PseudoResolutionError, match="dataset for O, not Si"):
        load_pseudopotentials(config, ["Si"], download=False)


def test_species_without_a_dataset_and_without_a_family():
    config = make_config(pseudo={"files": {"Si": "Si.pbe.psp8.gz"}})
    with pytest.raises(PseudoResolutionError, match=r"set pseudo\.family"):
        load_pseudopotentials(config, ["Si", "O"], download=False)


def test_unknown_family_is_reported():
    config = make_config(pseudo={"family": "not-a-family", "files": {}})
    with pytest.raises(PseudoResolutionError, match=r"unknown pseudo\.family"):
        load_pseudopotentials(config, ["Si"], download=False)


def test_family_kind_must_match_the_method():
    config = make_config(
        pseudo={"method": "paw", "family": "pseudodojo-nc-sr-pbe-standard-v0.4", "files": {}}
    )
    with pytest.raises(PseudoResolutionError, match="is a nc table"):
        load_pseudopotentials(config, ["Si"], download=False)


# ------------------------------------------------------------- XC consistency check
def test_xc_mismatch_is_an_error_by_default():
    """Using a PBE dataset in a PBEsol run is a real inconsistency, not a rounding detail."""
    config = make_config(xc={"functional": "PBEsol"})
    with pytest.raises(PseudoResolutionError) as excinfo:
        load_pseudopotentials(config, ["Si"], download=False)
    message = str(excinfo.value)
    assert "generated with PBE" in message
    assert "xc.functional is PBEsol" in message
    assert "allow_xc_mismatch" in message


def test_xc_mismatch_can_be_overridden_and_then_warns():
    config = make_config(
        xc={"functional": "PBEsol"},
        pseudo={"allow_xc_mismatch": True, "files": {"Si": "Si.pbe.psp8.gz"}},
    )
    resolved = load_pseudopotentials(config, ["Si"], download=False)
    assert len(resolved.warnings) == 1
    assert "allow_xc_mismatch is set" in resolved.warnings[0]


def test_matching_functional_produces_no_warning():
    config = make_config(xc={"functional": "PBEsol"}, pseudo={"files": {"Si": "Si.pbesol.psp8.gz"}})
    resolved = load_pseudopotentials(config, ["Si"], download=False)
    assert resolved.warnings == []
    assert resolved.pseudopotentials["Si"].xc_functional == "PBEsol"


def test_paw_dataset_rejected_when_method_is_nc(tmp_path, synthetic_upf_paw_file):
    """A PAW UPF read under pseudo.method: nc must be refused, not silently truncated.

    The file is a real Kresse-Joubert dataset, not a norm-conserving one relabelled: the
    reader now dispatches on the header, so a relabelled file would fail for the different
    and less interesting reason that it carries no augmentation block.
    """
    path = tmp_path / "C_paw.UPF"
    path.write_text(synthetic_upf_paw_file.read_text(encoding="utf-8"), encoding="utf-8")
    config = make_config(pseudo={"directory": str(tmp_path), "files": {"C": "C_paw.UPF"}})
    with pytest.raises(PseudoResolutionError, match="is a PAW dataset"):
        load_pseudopotentials(config, ["C"], download=False)


def test_a_relabelled_norm_conserving_file_says_what_is_missing(tmp_path):
    """Changing the header alone does not make a file a PAW dataset, and it must say so."""
    from tests.unit.test_pseudo_readers import _synthetic_upf

    text = _synthetic_upf().replace('pseudo_type="NC"', 'pseudo_type="PAW"')
    path = tmp_path / "Si_paw.upf"
    path.write_text(text, encoding="utf-8")
    config = make_config(pseudo={"directory": str(tmp_path), "files": {"Si": "Si_paw.upf"}})
    with pytest.raises(PseudoResolutionError, match="no PP_AUGMENTATION block"):
        load_pseudopotentials(config, ["Si"], download=False)


# ------------------------------------------------------------------------- cutoffs
def test_explicit_cutoff_is_used_and_converted():
    config = make_config()  # ecut: 60, a bare number meaning Rydberg
    resolved = load_pseudopotentials(config, ["Si"], download=False)
    ecut, ecut_rho = resolve_cutoffs(config, resolved)
    assert ecut == pytest.approx(30.0)  # Hartree
    assert ecut_rho == pytest.approx(120.0)  # 4x for norm-conserving


def test_explicit_cutoff_accepts_a_unit():
    config = make_config(basis={"ecut": "816 eV"})
    resolved = load_pseudopotentials(config, ["Si"], download=False)
    ecut, _ = resolve_cutoffs(config, resolved)
    assert ecut == pytest.approx(816 / 27.211386, rel=1e-6)


def test_cutoff_without_a_source_is_an_error():
    config = make_config(basis={"ecut": None}, pseudo={"files": {"Si": "Si.pbe.psp8.gz"}})
    resolved = load_pseudopotentials(config, ["Si"], download=False)
    with pytest.raises(PseudoResolutionError, match="carry no recommended cutoff"):
        resolve_cutoffs(config, resolved)


# ------------------------------------------------------------------------ manifests
@pytest.fixture
def cached_manifest(tmp_path):
    """A manifest cached on disk, so no network is needed."""
    family = KNOWN_FAMILIES["pseudodojo-nc-sr-pbe-standard-v0.4"]
    payload = {
        "pseudos_metadata": {
            "Si": {
                "basename": "Si.psp8",
                "Z_val": 4.0,
                "md5": "45c393525680abc7b25445b8ec62bf83",
                "hints": {"low": {"ecut": 14.0}, "normal": {"ecut": 18.0}, "high": {"ecut": 24.0}},
            },
            "Fe": {
                "basename": "Fe-sp.psp8",
                "Z_val": 16.0,
                "hints": {"low": {"ecut": 41.0}, "normal": {"ecut": 45.0}, "high": {"ecut": 53.0}},
            },
        }
    }
    cache = tmp_path / family.name / "manifest.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return family, tmp_path


def test_manifest_is_read_from_the_cache(cached_manifest):
    family, directory = cached_manifest
    manifest = load_manifest(family, directory, download=False)
    assert "Si" in manifest["pseudos_metadata"]


def test_missing_manifest_without_download_is_reported(tmp_path):
    family = KNOWN_FAMILIES["pseudodojo-nc-sr-pbe-standard-v0.4"]
    with pytest.raises(PseudoResolutionError, match="fetch-pseudos"):
        load_manifest(family, tmp_path, download=False)


def test_recommended_cutoff_takes_the_maximum_over_species(cached_manifest):
    """A cutoff converged for the hardest element is converged for all of them."""
    family, directory = cached_manifest
    assert recommended_cutoff(family, ["Si"], directory, download=False) == 18.0
    assert recommended_cutoff(family, ["Fe"], directory, download=False) == 45.0
    assert recommended_cutoff(family, ["Si", "Fe"], directory, download=False) == 45.0
    assert recommended_cutoff(family, ["Si"], directory, accuracy="high", download=False) == 24.0


def test_recommended_cutoff_rejects_an_unknown_element(cached_manifest):
    family, directory = cached_manifest
    with pytest.raises(PseudoResolutionError, match="no dataset for Xe"):
        recommended_cutoff(family, ["Xe"], directory, download=False)


@pytest.mark.physics
def test_fetching_a_real_dataset(tmp_path):
    """Downloads from PseudoDojo and checks the file against the manifest's md5."""
    from amaterasu.pseudo.registry import fetch_family

    paths = fetch_family("pseudodojo-nc-sr-pbesol-standard-v0.4", tmp_path, ["Si"])
    assert paths["Si"].exists()

    from amaterasu.pseudo.readers import read_pseudopotential

    pseudo = read_pseudopotential(paths["Si"])
    assert pseudo.xc_functional == "PBEsol"
    assert pseudo.check() == []
