"""Gates 2 and 3: every input problem is reported at once; --set reaches nested fields."""

from __future__ import annotations

import pytest
import yaml

from amaterasu.config import Config, ConfigError, config_to_yaml, load_config, load_config_mapping
from amaterasu.config._yaml import safe_load as yaml_safe_load
from amaterasu.config.overrides import apply_overrides, parse_override


def _minimal() -> dict:
    return {"structure": "Si.str", "basis": {"ecut": 60}}


def test_defaults_are_expanded():
    config = load_config_mapping(_minimal())
    assert config.solver == "scf"
    assert config.xc.functional == "PBE"
    assert config.scf.mixing.algorithm == "pulay"
    assert config.scf.mixing.preconditioner == "kerker"
    assert config.relax.algorithm == "fire"
    assert config.md.ensemble == "nvt"


def test_structure_is_required():
    with pytest.raises(ConfigError, match="required"):
        load_config_mapping({"basis": {"ecut": 60}})


def test_cutoff_is_never_guessed():
    """ecut is the primary convergence parameter and has no safe default."""
    with pytest.raises(ConfigError, match=r"basis\.ecut"):
        load_config_mapping({"structure": "Si.str"})
    # ...unless a dataset family is named, which carries a recommended value.
    load_config_mapping({"structure": "Si.str", "pseudo": {"family": "pseudodojo-pbe"}})


def test_unknown_key_is_reported_with_a_suggestion():
    with pytest.raises(ConfigError) as excinfo:
        load_config_mapping({**_minimal(), "kpoints": {"gird": [4, 4, 4]}})
    message = str(excinfo.value)
    assert "kpoints.gird" in message
    assert "did you mean 'grid'?" in message


def test_all_structural_and_semantic_problems_reported_together():
    """Gate 2: one pass must surface type errors and cross-field errors alike."""
    bad = {
        "structure": "Si.str",
        "solver": "relaxx",
        "xc": {"functional": "PBE0"},
        "basis": {"ecut": -5},
        "kpoints": {"grid": [8, 8]},
        "parallel": {"band_groups": 0},
    }
    with pytest.raises(ConfigError) as excinfo:
        load_config_mapping(bad, source="bad.yaml")
    problems = "\n".join(excinfo.value.problems)
    assert len(excinfo.value.problems) >= 5
    for expected in ("solver", "xc.functional", "basis.ecut", "kpoints.grid", "band_groups"):
        assert expected in problems, expected


def test_choice_matching_is_case_insensitive():
    config = load_config_mapping({**_minimal(), "xc": {"functional": "pbesol"}})
    assert config.xc.functional == "PBEsol"


def test_bool_is_not_a_number():
    """bool subclasses int, so accepting True as a cutoff would be a silent bug."""
    with pytest.raises(ConfigError):
        load_config_mapping({**_minimal(), "scf": {"max_steps": True}})


@pytest.mark.parametrize(
    ("assignment", "path", "value"),
    [
        ("basis.ecut=80", ["basis", "ecut"], 80),
        ("basis.ecut='80 Ry'", ["basis", "ecut"], "80 Ry"),
        ("kpoints.grid=[4, 4, 4]", ["kpoints", "grid"], [4, 4, 4]),
        ("relax.cell=true", ["relax", "cell"], True),
        ("occupations.nbands=null", ["occupations", "nbands"], None),
        # YAML 1.1 would make this a string; amaterasu resolves floats per YAML 1.2.
        ("scf.tol_energy=1e-8", ["scf", "tol_energy"], 1e-8),
        ("scf.tol_energy=1.0e-8", ["scf", "tol_energy"], 1e-8),
    ],
)
def test_parse_override(assignment, path, value):
    assert parse_override(assignment) == (path, value)


def test_yaml_12_floats_in_files():
    """PyYAML's YAML 1.1 would read 1e-6 as the string '1e-6'; amaterasu must not."""
    config = load_config_mapping(
        yaml_safe_load("structure: s\nbasis: {ecut: 60}\nscf: {tol_energy: 1e-8}\n")
    )
    assert config.scf.tol_energy == pytest.approx(1e-8)


def test_element_symbol_no_is_not_a_boolean():
    """YAML 1.1's yes/no booleans turn nobelium into False -- the Norway problem."""
    data = yaml_safe_load(
        "structure: s\nbasis: {ecut: 60}\n"
        "system: {hubbard: {scheme: dudarev, projector: atomic,"
        " species: {No: {U: 4.0, J: 0.5}}}}\n"
    )
    config = load_config_mapping(data)
    assert "No" in config.system.hubbard.species
    assert config.system.hubbard.species["No"].U == pytest.approx(4.0)


def test_yaml_11_boolean_words_get_a_helpful_message():
    with pytest.raises(ConfigError, match="true/false"):
        load_config_mapping(
            yaml_safe_load("structure: s\nbasis: {ecut: 60}\nkpoints: {symmetry: yes}\n")
        )


def test_overrides_reach_nested_fields():
    """Gate 3: dotted paths must reach arbitrarily deep, including new dict entries."""
    config = load_config_mapping(
        _minimal(),
        overrides=[
            "kpoints.grid=[4,4,4]",
            "scf.mixing.beta=0.2",
            "system.hubbard.scheme=dudarev",
            "system.hubbard.species.Ni.U=6.0",
            "system.hubbard.species.Ni.J=1.0",
            "system.hubbard.projector=atomic",
        ],
    )
    assert config.kpoints.grid == [4, 4, 4]
    assert config.scf.mixing.beta == pytest.approx(0.2)
    assert config.system.hubbard.species["Ni"].U == pytest.approx(6.0)
    assert config.system.hubbard.species["Ni"].J == pytest.approx(1.0)


def test_overrides_are_validated_like_the_file():
    with pytest.raises(ConfigError, match=r"scf\.mixing\.beta"):
        load_config_mapping(_minimal(), overrides=["scf.mixing.beta=5"])


def test_override_through_a_scalar_is_an_error():
    with pytest.raises(ValueError, match="not a section"):
        apply_overrides({"solver": "scf"}, ["solver.sub=1"])


@pytest.mark.parametrize("bad", ["novalue", "=5", " =x"])
def test_malformed_override(bad):
    with pytest.raises(ValueError):
        parse_override(bad)


# --------------------------------------------------------------- semantic checks
def test_density_cutoff_may_not_be_below_the_wavefunction_cutoff():
    with pytest.raises(ConfigError, match="ecut_rho"):
        load_config_mapping({"structure": "s", "basis": {"ecut": 60, "ecut_rho": 30}})


def test_hubbard_scheme_and_species_must_agree():
    with pytest.raises(ConfigError, match="species"):
        load_config_mapping({**_minimal(), "system": {"hubbard": {"scheme": "dudarev"}}})
    with pytest.raises(ConfigError, match="scheme"):
        load_config_mapping({**_minimal(), "system": {"hubbard": {"species": {"Ni": {"U": 6.0}}}}})


def test_negative_u_eff_is_rejected_under_dudarev():
    """U_eff = U - J < 0 localises the shell in the wrong direction."""
    with pytest.raises(ConfigError, match="U_eff"):
        load_config_mapping(
            {
                **_minimal(),
                "system": {
                    "hubbard": {
                        "scheme": "dudarev",
                        "projector": "atomic",
                        "species": {"Ni": {"U": 1.0, "J": 4.0}},
                    }
                },
            }
        )


def test_paw_projector_requires_a_paw_dataset():
    with pytest.raises(ConfigError, match="projector"):
        load_config_mapping(
            {
                **_minimal(),
                "pseudo": {"method": "nc"},
                "system": {
                    "hubbard": {
                        "scheme": "dudarev",
                        "projector": "paw",
                        "species": {"Ni": {"U": 6.0, "J": 1.0}},
                    }
                },
            }
        )


def test_pbesol_paw_requires_a_upf_family():
    """The PAW-XML tables (JTH, GPAW) ship PBE and LDA only; PBEsol PAW is pslibrary UPF."""
    with pytest.raises(ConfigError, match="PBEsol"):
        load_config_mapping(
            {
                **_minimal(),
                "xc": {"functional": "PBEsol"},
                "pseudo": {"method": "paw", "family": "jth-pbe-v2.0"},
            }
        )
    load_config_mapping(
        {
            **_minimal(),
            "xc": {"functional": "PBEsol"},
            "pseudo": {"method": "paw", "family": "pslibrary-pbesol-kjpaw-1.0.0"},
        }
    )


def test_neb_requires_two_endpoints():
    with pytest.raises(ConfigError) as excinfo:
        load_config_mapping({**_minimal(), "solver": "neb"})
    problems = "\n".join(excinfo.value.problems)
    assert "neb.initial" in problems
    assert "neb.final" in problems


def test_nve_accepts_a_temperature_for_the_initial_velocities():
    """NVE has to start the atoms moving somehow; it just does not control T afterwards."""
    config = load_config_mapping(
        {**_minimal(), "solver": "md", "md": {"ensemble": "nve", "temperature": 300}}
    )
    assert config.md.temperature == 300.0

    with pytest.raises(ConfigError, match="must be positive"):
        load_config_mapping(
            {**_minimal(), "solver": "md", "md": {"ensemble": "nve", "temperature": 0}}
        )


def test_tetrahedron_needs_a_mesh():
    with pytest.raises(ConfigError, match="tetrahedron"):
        load_config_mapping(
            {
                **_minimal(),
                "kpoints": {"scheme": "gamma"},
                "occupations": {"scheme": "tetrahedron"},
            }
        )


# --------------------------------------------------------------- serialisation
def test_resolved_yaml_round_trips(si_config_text):
    config = load_config_mapping(yaml.safe_load(si_config_text))
    text = config_to_yaml(config)
    again = load_config_mapping(yaml.safe_load(text))
    assert config_to_yaml(again) == text


def test_resolved_yaml_contains_every_default(si_config_text):
    """resolved_input.yaml must record settings the user never wrote, for reproducibility."""
    data = yaml.safe_load(config_to_yaml(load_config_mapping(yaml.safe_load(si_config_text))))
    assert data["scf"]["mixing"]["algorithm"] == "pulay"
    assert data["parallel"]["fft_backend"] == "scipy"
    assert data["output"]["log_level"] == "info"


def test_load_from_file(tmp_path, si_config_text):
    path = tmp_path / "calc.yaml"
    path.write_text(si_config_text, encoding="utf-8")
    assert isinstance(load_config(path), Config)


def test_empty_and_invalid_files(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(empty)

    broken = tmp_path / "broken.yaml"
    broken.write_text("structure: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(broken)


def test_direct_minimisation_requires_fixed_occupations():
    """Catch the combination at validation time, not after the basis has been built."""
    base = {"structure": "Si.str", "basis": {"ecut": 40}}
    with pytest.raises(ConfigError, match="minimises the energy over the orbitals"):
        load_config_mapping(
            {
                **base,
                "scf": {"method": "direct-lbfgs"},
                "occupations": {"scheme": "gaussian"},
            }
        )
    # The supported combination validates cleanly.
    config = load_config_mapping(
        {**base, "scf": {"method": "direct-lbfgs"}, "occupations": {"scheme": "fixed"}}
    )
    assert config.scf.method == "direct-lbfgs"
