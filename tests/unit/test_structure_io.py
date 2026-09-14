"""Gate 1: the structure reader reproduces Si's known properties, in every format."""

from __future__ import annotations

import numpy as np
import pytest
from ase.io import write as ase_write

from amaterasu.io.structure_read import (
    StructureParseError,
    looks_like_amaterasu_format,
    read_amaterasu_structure,
    read_structure,
)
from amaterasu.io.structure_write import format_amaterasu_structure, write_amaterasu_structure
from amaterasu.system.structure import summarize
from amaterasu.units import BOHR_IN_ANG

from ..conftest import SI_REFERENCE


def test_si_reference_properties(si_file):
    """Every number here is known independently: closed form, or experiment."""
    summary = summarize(read_structure(si_file))
    assert summary.formula == "Si2"
    assert summary.nearest_distance == pytest.approx(SI_REFERENCE["nearest_neighbour"], abs=1e-4)
    assert summary.volume == pytest.approx(SI_REFERENCE["volume"], rel=1e-9)
    assert summary.density == pytest.approx(SI_REFERENCE["density"], abs=1e-3)
    assert summary.symmetry.number == SI_REFERENCE["spacegroup"]
    assert summary.symmetry.n_operations == SI_REFERENCE["n_operations"]
    assert summary.symmetry.n_inequivalent_atoms == 1
    assert summary.warnings == []


def test_agrees_with_ase_built_structure(si_file, si_atoms):
    """The parsed cell must match one ASE builds from the same lattice constant."""
    parsed = read_structure(si_file)
    assert np.allclose(parsed.cell[:], si_atoms.cell[:], atol=1e-9)
    assert np.allclose(parsed.get_scaled_positions(), si_atoms.get_scaled_positions(), atol=1e-12)


@pytest.mark.parametrize("fmt", ["vasp", "cif", "extxyz"])
def test_matches_ase_formats(tmp_path, si_file, fmt):
    """The same structure read from POSCAR/CIF/extxyz gives the same summary."""
    native = read_structure(si_file)
    path = tmp_path / f"si.{fmt}"
    ase_write(path, native, format=fmt)
    other = read_structure(path)
    assert other.get_chemical_formula() == native.get_chemical_formula()
    assert other.get_volume() == pytest.approx(native.get_volume(), rel=1e-5)
    a = summarize(native)
    b = summarize(other)
    assert b.nearest_distance == pytest.approx(a.nearest_distance, abs=1e-4)
    assert b.symmetry.number == a.symmetry.number


def test_round_trip_is_lossless(tmp_path):
    """Cell, positions, magnetic moments and constraints all survive a write/read cycle."""
    text = """
    units bohr
    %block lattice_vectors
      0.0      5.13116  5.13116
      5.13116  0.0      5.13116
      5.13116  5.13116  0.0
    %endblock lattice_vectors
    %block atomic_positions cartesian
      Si 0.0     0.0     0.0                             fix=xyz
      Si 2.56558 2.56558 2.56558   magmom=0.5  fix=z
    %endblock atomic_positions
    """
    original = read_amaterasu_structure("<inline>", text=text)
    path = tmp_path / "round.str"
    write_amaterasu_structure(original, path)
    again = read_structure(path)

    assert np.allclose(original.cell[:], again.cell[:], atol=1e-9)
    assert np.allclose(original.get_positions(), again.get_positions(), atol=1e-9)
    assert np.allclose(
        original.get_initial_magnetic_moments(), again.get_initial_magnetic_moments()
    )
    assert len(again.constraints) == len(original.constraints)


def test_bohr_units_are_converted(si_text):
    """The same cell given in Bohr must come out identical in Angstrom."""
    angstrom = read_amaterasu_structure("<ang>", text=si_text)
    bohr_text = si_text.replace("units       angstrom", "units       bohr").replace(
        "2.71535", f"{2.71535 / BOHR_IN_ANG:.10f}"
    )
    bohr = read_amaterasu_structure("<bohr>", text=bohr_text)
    assert np.allclose(angstrom.cell[:], bohr.cell[:], atol=1e-8)


def test_scale_multiplies_the_lattice(si_text):
    scaled = read_amaterasu_structure("<s>", text=si_text.replace("scale       1.0", "scale 2.0"))
    plain = read_amaterasu_structure("<p>", text=si_text)
    assert scaled.get_volume() == pytest.approx(8 * plain.get_volume(), rel=1e-12)


def test_format_detection(si_text):
    assert looks_like_amaterasu_format(si_text)
    assert not looks_like_amaterasu_format("Si2\nCell: 5.43\n")


def test_every_error_is_reported_in_one_pass():
    """Gate 2 for structures: one exception listing every problem, in line order."""
    broken = """
units       furlongs
scale       -1.0
bogus_key   3

%block lattice_vectors
  1.0  0.0  0.0
  0.0  1.0
%endblock lattice_vectors

%block atomic_positions
   SI    0.0  0.0  0.0
   Xx    0.1  0.1  0.1
   Si    0.2  0.2  0.2   fix=q  charge=1
%endblock atomic_positions
"""
    with pytest.raises(StructureParseError) as excinfo:
        read_amaterasu_structure("<broken>", text=broken)

    problems = excinfo.value.problems
    assert len(problems) >= 9
    line_numbers = [lineno for lineno, _ in problems if lineno is not None]
    assert line_numbers == sorted(line_numbers), "problems must be reported in file order"

    text = str(excinfo.value)
    for expected in (
        "furlongs",
        "scale",
        "bogus_key",
        "coordinate type",
        "SI",
        "Xx",
        "fix=",
        "charge",
    ):
        assert expected in text, expected


def test_coordinate_type_is_never_defaulted():
    """Omitting fractional/cartesian is an error, not a silent guess."""
    text = """
%block lattice_vectors
 1 0 0
 0 1 0
 0 0 1
%endblock lattice_vectors
%block atomic_positions
 Si 0.0 0.0 0.0
%endblock atomic_positions
"""
    with pytest.raises(StructureParseError, match="coordinate type"):
        read_amaterasu_structure("<no-modifier>", text=text)


def test_singular_cell_is_rejected():
    text = """
%block lattice_vectors
 1 0 0
 2 0 0
 0 0 1
%endblock lattice_vectors
%block atomic_positions fractional
 Si 0.0 0.0 0.0
%endblock atomic_positions
"""
    with pytest.raises(StructureParseError, match=r"singular|degenerate"):
        read_amaterasu_structure("<singular>", text=text)


def test_implausible_distance_warns():
    """A Bohr/Angstrom mix-up shows up as an impossibly short bond, not a silent result."""
    text = """
%block lattice_vectors
 3 0 0
 0 3 0
 0 0 3
%endblock lattice_vectors
%block atomic_positions cartesian
 Si 0.0 0.0 0.0
 Si 0.2 0.0 0.0
%endblock atomic_positions
"""
    summary = summarize(read_amaterasu_structure("<close>", text=text))
    assert any("nearest-neighbour" in w for w in summary.warnings)


def test_non_periodic_input_is_rejected(tmp_path):
    """amaterasu is a 3D-periodic code; a molecule without a cell must not be invented one."""
    from ase import Atoms

    path = tmp_path / "mol.xyz"
    ase_write(path, Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]), format="xyz")
    with pytest.raises(ValueError, match="3D-periodic"):
        read_structure(path)


def test_written_file_is_parseable(si_file):
    atoms = read_structure(si_file)
    text = format_amaterasu_structure(atoms, comment="test")
    assert looks_like_amaterasu_format(text)
    assert "fractional" in text
    reparsed = read_amaterasu_structure("<written>", text=text)
    assert np.allclose(reparsed.cell[:], atoms.cell[:], atol=1e-9)
