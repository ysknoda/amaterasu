"""Gate 7: unit constants agree with ase.units exactly and with CODATA closely."""

from __future__ import annotations

import math

import pytest
import scipy.constants as sc
from ase import units as ase_units

from amaterasu import units


def test_derived_from_ase_exactly():
    """The single source of truth is ase.units, so agreement must be exact."""
    assert units.HARTREE_IN_EV == ase_units.Ha
    assert units.BOHR_IN_ANG == ase_units.Bohr
    assert units.RYDBERG_IN_EV == ase_units.Ry


def test_rydberg_is_half_a_hartree():
    assert units.RYDBERG_IN_HARTREE == pytest.approx(0.5, abs=1e-14)


@pytest.mark.parametrize(
    ("value", "codata", "name"),
    [
        (lambda: units.HARTREE_IN_EV, lambda: sc.value("Hartree energy in eV"), "Hartree"),
        (lambda: units.BOHR_IN_ANG, lambda: sc.value("Bohr radius") * 1e10, "Bohr radius"),
    ],
)
def test_matches_independent_codata(value, codata, name):
    """Cross-check against scipy's CODATA 2022.

    ASE 3.28 ships CODATA 2014, so a 1e-12 comparison would fail on the revision alone.
    A 1e-7 tolerance still catches a genuinely wrong constant.
    """
    assert value() == pytest.approx(codata(), rel=1e-7), name


def _codata_reference() -> dict[str, float]:
    """Atomic units derived from scipy's CODATA 2022, computed rather than quoted.

    Hard-coding these invites a transcription error that a test would then enshrine, so
    they are built from the two primitives (the Hartree in eV and the Bohr radius) here.
    """
    hartree_ev = sc.value("Hartree energy in eV")
    bohr_ang = sc.value("Bohr radius") * 1e10
    return {
        "force": hartree_ev / bohr_ang,  # eV/Angstrom
        "pressure": hartree_ev / bohr_ang**3 * sc.e * 1e30 / 1e9,  # GPa
        "kb": sc.value("Boltzmann constant in eV/K") / hartree_ev,  # Ha/K
        "time": sc.value("atomic unit of time"),  # s
    }


def test_atomic_unit_derived_values():
    """Derived atomic units against CODATA 2022.

    The tolerance is 1e-6 rather than machine precision because amaterasu inherits ASE's
    CODATA 2014 values; the largest resulting discrepancy is in the Boltzmann constant
    (relative 3.8e-7). Anything looser would stop catching a genuinely wrong constant.
    """
    reference = _codata_reference()
    assert units.HARTREE_PER_BOHR_IN_EV_PER_ANG == pytest.approx(reference["force"], rel=1e-6)
    assert units.HARTREE_PER_BOHR3_IN_GPA == pytest.approx(reference["pressure"], rel=1e-6)
    assert units.KB_HARTREE_PER_K == pytest.approx(reference["kb"], rel=1e-6)
    assert units.ATOMIC_TIME_IN_SECONDS == pytest.approx(reference["time"], rel=1e-6)


def test_output_unit_convention():
    """Reported quantities use eV, Angstrom, eV/Angstrom, eV/Angstrom^3, GPa and fs (ADR-0014)."""
    assert units.OUTPUT_UNITS == {
        "energy": "eV",
        "length": "Angstrom",
        "force": "eV/Angstrom",
        "stress": "eV/Angstrom^3",
        "pressure": "GPa",
        "time": "fs",
    }
    assert units.STRESS_CONVENTION == "positive_under_compression"
    reference = _codata_reference()
    assert units.energy_to_output(1.0) == pytest.approx(sc.value("Hartree energy in eV"), rel=1e-6)
    assert units.length_to_output(1.0) == pytest.approx(sc.value("Bohr radius") * 1e10, rel=1e-6)
    assert units.force_to_output(1.0) == pytest.approx(reference["force"], rel=1e-6)
    assert units.pressure_to_output(1.0) == pytest.approx(reference["pressure"], rel=1e-6)
    # One femtosecond expressed in atomic time units must come back as 1 fs.
    one_fs_in_au = units.FS_IN_SECONDS / units.ATOMIC_TIME_IN_SECONDS
    assert units.time_to_output(one_fs_in_au) == pytest.approx(1.0, rel=1e-12)
    # Stress: eV/Angstrom^3, sign flipped from the internal (ASE) convention.
    assert units.stress_to_output(1.0) == pytest.approx(
        -reference["pressure"] * units.GPA_IN_EV_PER_ANG3, rel=1e-6
    )


@pytest.mark.parametrize(
    ("value", "expected_hartree"),
    [
        (60, 30.0),  # bare number -> Ry, the pslibrary/QE convention
        ("60 Ry", 30.0),
        ("60Ry", 30.0),
        ("30 Ha", 30.0),
        ("30 hartree", 30.0),
        ("1.0d1 Ha", 10.0),  # Fortran-style exponent
    ],
)
def test_parse_energy(value, expected_hartree):
    assert units.parse_energy(value) == pytest.approx(expected_hartree, rel=1e-12)


def test_parse_energy_ev():
    assert units.parse_energy("27.211386024367243 eV") == pytest.approx(1.0, rel=1e-12)
    assert units.parse_energy(816, "eV") == pytest.approx(816 / units.HARTREE_IN_EV)


def test_parse_length():
    assert units.parse_length("1 Bohr") == pytest.approx(1.0)
    assert units.parse_length(1.0) == pytest.approx(units.ANG_IN_BOHR)


@pytest.mark.parametrize("bad", ["60 furlongs", "not a number", "", "60 eV extra"])
def test_parse_energy_rejects_garbage(bad):
    with pytest.raises(ValueError):
        units.parse_energy(bad)


def test_parse_energy_rejects_bool():
    """bool is an int subclass; silently reading True as 1 Ry would be a real bug."""
    with pytest.raises(TypeError):
        units.parse_energy(True)


def test_no_nan_or_inf():
    for name in units.__all__:
        value = getattr(units, name)
        if isinstance(value, float):
            assert math.isfinite(value), name
