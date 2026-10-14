"""End-to-end checks of the command line interface."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from amaterasu.cli import main
from amaterasu.units import GPA_IN_EV_PER_ANG3

from ..conftest import SI_REFERENCE


@pytest.fixture
def project(tmp_path, si_text, si_config_text):
    """A directory holding a structure file and a calculation input that refers to it."""
    (tmp_path / "Si.str").write_text(si_text, encoding="utf-8")
    (tmp_path / "calc.yaml").write_text(si_config_text, encoding="utf-8")
    return tmp_path


def test_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "amaterasu" in capsys.readouterr().out


def test_info_structure_reports_the_reference_numbers(project, capsys):
    assert main(["info", str(project / "Si.str")]) == 0
    out = capsys.readouterr().out
    assert "Si2" in out
    assert "Fd-3m (227)" in out
    assert f"{SI_REFERENCE['volume']:.6f}" in out
    assert f"{SI_REFERENCE['nearest_neighbour']:.6f}" in out
    assert "symprec sensitivity" in out


def test_info_states_the_output_units(project, capsys):
    main(["info", str(project / "Si.str")])
    out = capsys.readouterr().out
    assert "energy eV" in out
    assert "length Angstrom" in out
    assert "force eV/Angstrom" in out
    assert "stress eV/Angstrom^3" in out
    assert "pressure GPa" in out
    assert "MD time fs" in out


def test_info_on_a_calculation_input(project, capsys):
    """info resolves the pseudopotentials and derives the density cutoff from them."""
    assert main(["info", str(project / "calc.yaml"), "--offline"]) == 0
    out = capsys.readouterr().out
    assert "8x8x8 mesh, 512 points -> 29 in the irreducible wedge" in out
    assert "pseudopotentials for Si" in out
    assert "Z_val=4" in out
    assert "valence electrons  4" in out
    assert "ecut               30 Ha = 60 Ry" in out
    assert "ecut_rho           120 Ha = 240 Ry" in out


def test_info_refuses_a_functional_the_dataset_was_not_made_for(project, capsys):
    """A PBE dataset in a PBEsol run is an inconsistency, and must stop the run."""
    code = main(["info", str(project / "calc.yaml"), "--offline", "--set", "xc.functional=PBEsol"])
    assert code == 2
    err = capsys.readouterr().err
    assert "generated with PBE" in err
    assert "allow_xc_mismatch" in err


def test_families_lists_the_known_tables(capsys):
    assert main(["families"]) == 0
    out = capsys.readouterr().out
    assert "pseudodojo-nc-sr-pbesol-standard-v0.4" in out
    assert "CC BY 4.0" in out
    # PBEsol PAW is not fetchable and the listing must say so rather than omit it silently.
    assert "PBEsol PAW datasets are not listed" in out


def test_fetch_pseudos_rejects_an_unknown_family(capsys, tmp_path):
    code = main(["fetch-pseudos", "--family", "nope", "-d", str(tmp_path)])
    assert code == 2
    assert "unknown family" in capsys.readouterr().err


def test_dry_run_validates_without_writing(project, capsys):
    assert main(["run", str(project / "calc.yaml"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run: no calculation performed" in out
    assert "resolved input" in out
    assert not (project / "out").exists(), "--dry-run must not create anything"


def test_dry_run_prints_expanded_defaults(project, capsys):
    main(["run", str(project / "calc.yaml"), "--dry-run"])
    out = capsys.readouterr().out
    body = out.split("--- resolved input (all defaults expanded) ---", 1)[1]
    data = yaml.safe_load(body.split("--- dry run", 1)[0])
    assert data["scf"]["mixing"]["algorithm"] == "pulay"
    assert data["relax"]["cell_filter"] == "frechet"


def test_overrides_from_the_command_line(project, capsys):
    main(
        [
            "run",
            str(project / "calc.yaml"),
            "--dry-run",
            "--set",
            "kpoints.grid=[4,4,4]",
            "--set",
            "system.spin=polarized",
            "--set",
            "basis.ecut='816 eV'",
        ]
    )
    out = capsys.readouterr().out
    assert "4x4x4 mesh, 64 points -> 8 in the irreducible wedge" in out
    assert "spin                 polarized" in out
    assert "816 eV" in out


def test_structure_path_is_relative_to_the_input_file(project, capsys, tmp_path, monkeypatch):
    """A run started from another directory must still find the structure."""
    monkeypatch.chdir(tmp_path.parent)
    assert main(["run", str(project / "calc.yaml"), "--dry-run"]) == 0
    assert "Si2" in capsys.readouterr().out


def test_structure_override(project, tmp_path, si_text, capsys):
    other = tmp_path / "other.str"
    other.write_text(si_text.replace("2.71535", "2.80000"), encoding="utf-8")
    main(["run", str(project / "calc.yaml"), "--dry-run", "--structure", str(other)])
    assert "other.str" in capsys.readouterr().out


def test_run_writes_the_resolved_input(project, capsys):
    """Reproducibility: the fully expanded settings are recorded next to the results."""
    code = main(
        [
            "run",
            str(project / "calc.yaml"),
            "-o",
            str(project / "out"),
            "--set",
            "solver=dfpt",
        ]
    )
    assert code == 3, "DFPT lands in Phase 6, so the run reports that clearly"
    resolved = project / "out" / "resolved_input.yaml"
    assert resolved.exists()
    data = yaml.safe_load(resolved.read_text())
    assert data["basis"]["ecut"] == 60.0
    assert data["output"]["log_level"] == "info"
    assert "not implemented yet" in capsys.readouterr().out


def test_run_performs_an_scf_and_writes_hdf5(project, capsys):
    """The whole path from a YAML file to a result file, on a deliberately tiny basis."""
    code = main(
        [
            "run",
            str(project / "calc.yaml"),
            "-o",
            str(project / "out"),
            "--set",
            "solver=scf",
            "--set",
            "basis.ecut=12",
            "--set",
            "kpoints.grid=[2,2,2]",
            "--set",
            "occupations.scheme=fixed",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "SCF converged" in out
    assert "total energy E" in out
    assert "band gap" in out

    import h5py

    with h5py.File(project / "out" / "amaterasu.h5") as handle:
        assert handle.attrs["converged"]
        assert handle.attrs["functional"] == "PBE"
        assert handle["scf/eigenvalues"].shape[0] == 1
        assert handle["density"].attrs["units"] == "electrons/Angstrom^3"
        assert "resolved_input" in handle.attrs
        # ADR-0014: forces and stress are written, and the file says which sign it uses.
        assert handle.attrs["format_version"] == 2
        assert handle.attrs["stress_convention"] == "positive_under_compression"
        assert handle["forces"].shape == (2, 3)
        assert handle["forces"].attrs["units"] == "eV/Angstrom"
        stress = handle["stress"][()]
        assert stress.shape == (3, 3)
        assert handle["stress"].attrs["units"] == "eV/Angstrom^3"
        pressure = handle["stress"].attrs["pressure_GPa"]
        assert np.trace(stress) / 3 / GPA_IN_EV_PER_ANG3 == pytest.approx(pressure, rel=1e-12)
    assert "stress (GPa, positive under compression)" in out


def test_invalid_input_reports_every_problem(project, capsys):
    code = main(
        [
            "run",
            str(project / "calc.yaml"),
            "--dry-run",
            "--set",
            "solver=relaxx",
            "--set",
            "basis.ecut=-5",
            "--set",
            "kpoints.grid=[8,8]",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "solver" in err
    assert "basis.ecut" in err
    assert "kpoints.grid" in err


def test_broken_structure_reports_every_problem(tmp_path, capsys):
    path = tmp_path / "bad.str"
    path.write_text(
        "units furlongs\n"
        "%block lattice_vectors\n1 0 0\n0 1 0\n%endblock lattice_vectors\n"
        "%block atomic_positions\nSI 0 0 0\n%endblock atomic_positions\n",
        encoding="utf-8",
    )
    assert main(["info", str(path)]) == 2
    err = capsys.readouterr().err
    assert "furlongs" in err
    assert "coordinate type" in err


def test_convert_round_trip(project, tmp_path, capsys):
    poscar = tmp_path / "POSCAR"
    assert main(["convert", str(project / "Si.str"), "-o", str(poscar), "--to-format", "vasp"]) == 0
    back = tmp_path / "back.str"
    assert main(["convert", str(poscar), "-o", str(back)]) == 0
    assert "%block lattice_vectors" in back.read_text()

    capsys.readouterr()
    main(["info", str(back)])
    assert "Fd-3m (227)" in capsys.readouterr().out


def test_missing_file(capsys):
    assert main(["info", "does-not-exist.str"]) == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["plot", "bands", "out.h5"]])
def test_unimplemented_commands_say_so(argv, capsys):
    assert main(argv) == 3
    assert "not implemented yet" in capsys.readouterr().err


def test_convergence_scan_reports_a_settling_energy(project, capsys):
    """The scan must vary one parameter, state what it held fixed, and show the trend."""
    code = main(
        [
            "convergence",
            str(project / "calc.yaml"),
            "--scan",
            "basis.ecut",
            "--range",
            "8:16:4",
            "--offline",
            "--set",
            "kpoints.grid=[2,2,2]",
            "--set",
            "occupations.scheme=fixed",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "held fixed           kpoints monkhorst-pack [2, 2, 2]" in out
    assert "never both at once" in out
    assert "differences from the finest point" in out
    # The energy must fall monotonically as the basis grows -- it is variational.
    energies = [
        float(line.split()[1])
        for line in out.splitlines()
        if line.startswith("  ") and len(line.split()) == 5 and line.split()[0] in ("8", "12", "16")
    ]
    assert len(energies) == 3
    assert energies[0] > energies[1] > energies[2]


def test_convergence_scan_of_the_k_point_mesh(project, capsys):
    """A scan of kpoints.grid means an n x n x n mesh, which is what a k-scan varies."""
    code = main(
        [
            "convergence",
            str(project / "calc.yaml"),
            "--scan",
            "kpoints.grid",
            "--range",
            "1:3:1",
            "--offline",
            "--set",
            "basis.ecut=12",
            "--set",
            "occupations.scheme=fixed",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "held fixed           basis.ecut 12" in out


def test_convergence_rejects_a_malformed_range(project, capsys):
    assert (
        main(
            ["convergence", str(project / "calc.yaml"), "--scan", "basis.ecut", "--range", "10:20"]
        )
        == 2
    )
    assert "START:STOP:STEP" in capsys.readouterr().err


def test_restart_reuses_a_previous_density(project, capsys):
    """A checkpoint of the same basis is picked up; a mismatched one is refused clearly."""
    common = [
        "run",
        str(project / "calc.yaml"),
        "-o",
        str(project / "out"),
        "--set",
        "solver=scf",
        "--set",
        "basis.ecut=12",
        "--set",
        "kpoints.grid=[2,2,2]",
        "--set",
        "occupations.scheme=fixed",
    ]
    assert main(common) == 0
    first = capsys.readouterr().out
    checkpoint = str(project / "out" / "amaterasu.h5")

    assert main([*common, "--restart", checkpoint]) == 0
    restarted = capsys.readouterr().out
    assert f"restarting from the density in {checkpoint}" in restarted
    # Starting from the converged density must take fewer iterations than starting cold.
    assert _iterations(restarted) < _iterations(first)

    # A different cutoff means a different grid, so the checkpoint cannot be used.
    assert main([*common[:-6], "--set", "basis.ecut=20", "--restart", checkpoint]) == 0
    assert "starting from atomic densities instead" in capsys.readouterr().out


def _iterations(output: str) -> int:
    for line in output.splitlines():
        if line.startswith("SCF converged in "):
            return int(line.split()[3])
    raise AssertionError("no convergence line in the output")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["solver=relax"], "only implemented for solver 'scf'"),
        (["solver=md"], "only implemented for solver 'scf'"),
        (["solver=bands"], "only implemented for solver 'scf'"),
        (["scf.method=direct-lbfgs", "occupations.scheme=fixed"], "needs scf.method: mixing"),
    ],
)
def test_restart_is_refused_where_it_would_be_ignored(project, capsys, overrides, message):
    """--restart used to be dropped silently by every solver but scf; refuse it up front,
    before any output is written."""
    argv = ["run", str(project / "calc.yaml"), "-o", str(project / "out"), "--dry-run"]
    for override in overrides:
        argv += ["--set", override]
    assert main([*argv, "--restart", str(project / "missing.h5")]) == 2
    assert message in capsys.readouterr().err
    assert not (project / "out").exists()


def test_log_level_warning_keeps_only_warnings(project, capsys):
    common = [
        "run",
        str(project / "calc.yaml"),
        "-o",
        str(project / "out"),
        "--set",
        "basis.ecut=12",
        "--set",
        "kpoints.grid=[2,2,2]",
        "--set",
        "occupations.scheme=fixed",
    ]
    assert main([*common, "--set", "output.log_level=warning"]) == 0
    quiet = capsys.readouterr().out
    assert "SCF converged" not in quiet
    assert "solver" not in quiet
    assert all(line.lstrip().startswith(("WARNING", "NOTE")) for line in quiet.splitlines())
    assert (project / "out" / "amaterasu.h5").exists(), "the results are written regardless"

    assert main([*common, "--set", "output.log_level=error"]) == 0
    assert capsys.readouterr().out == ""
