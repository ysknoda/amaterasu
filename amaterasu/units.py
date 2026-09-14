"""Single source of truth for units and physical constants in amaterasu.

Internal unit system
--------------------
amaterasu works internally in **Hartree atomic units**: energies in Hartree (Ha),
lengths in Bohr (a0), masses in electron masses. This is the natural system for the
Kohn-Sham equations (the kinetic operator is simply -1/2 nabla^2) and it is the system
used by the PAW-XML dataset format and by ABINIT psp8 files. UPF files use Rydberg
atomic units instead, so the UPF reader converts on load.

Conversion happens at exactly two boundaries:

    * the ASE interface (``amaterasu.calculator``) -- ASE speaks eV and Angstrom;
    * user input/output (YAML, log files, plots) -- see :func:`parse_energy`.

Provenance of the constants
---------------------------
Every constant here is **derived from** :mod:`ase.units` rather than from an independent
CODATA table. That is deliberate: ASE is a hard dependency and every structure, force and
stress crosses the ASE boundary, so deriving from a second source would introduce a silent
inconsistency at that boundary. ASE 3.28 ships CODATA 2014, which differs from CODATA 2022
by a relative 8.1e-9 in the Hartree and 3.7e-11 in the Bohr radius -- far below any
physically meaningful DFT error, but large enough to break a naive 1e-12 equality test.

``tests/unit/test_units.py`` checks these values against independent CODATA 2022 values
from :mod:`scipy.constants` at a 1e-7 tolerance, which catches a genuinely wrong constant
while tolerating the CODATA revision.
"""

from __future__ import annotations

import re

from ase import units as _ase

__all__ = [
    "ANG_IN_BOHR",
    "BOHR_IN_ANG",
    "CODATA_VERSION",
    "ENERGY_UNITS",
    "EV_IN_HARTREE",
    "GPA_IN_EV_PER_ANG3",
    "HARTREE_IN_EV",
    "HARTREE_PER_BOHR3_IN_GPA",
    "HARTREE_PER_BOHR_IN_EV_PER_ANG",
    "KB_HARTREE_PER_K",
    "LENGTH_UNITS",
    "RYDBERG_IN_EV",
    "RYDBERG_IN_HARTREE",
    "parse_energy",
    "parse_length",
]

#: CODATA revision underlying every constant in this module (inherited from ``ase.units``).
CODATA_VERSION: str = getattr(_ase, "__codata_version__", "unknown")

# --------------------------------------------------------------------------- energy
#: Electronvolts per Hartree.
HARTREE_IN_EV: float = float(_ase.Ha)
#: Hartree per electronvolt.
EV_IN_HARTREE: float = 1.0 / HARTREE_IN_EV
#: Electronvolts per Rydberg.
RYDBERG_IN_EV: float = float(_ase.Ry)
#: Hartree per Rydberg (exactly 1/2 by definition; asserted below).
RYDBERG_IN_HARTREE: float = RYDBERG_IN_EV / HARTREE_IN_EV

# --------------------------------------------------------------------------- length
#: Angstrom per Bohr.
BOHR_IN_ANG: float = float(_ase.Bohr)
#: Bohr per Angstrom.
ANG_IN_BOHR: float = 1.0 / BOHR_IN_ANG

# --------------------------------------------------------------------------- derived
#: eV/Angstrom per Ha/Bohr (atomic unit of force ~= 51.42 eV/A).
HARTREE_PER_BOHR_IN_EV_PER_ANG: float = HARTREE_IN_EV / BOHR_IN_ANG
#: GPa per Ha/Bohr^3 (atomic unit of pressure ~= 29421 GPa = 29.42 TPa).
HARTREE_PER_BOHR3_IN_GPA: float = HARTREE_IN_EV / BOHR_IN_ANG**3 / _ase.GPa
#: Boltzmann constant in Ha/K (~= 3.1668e-6).
KB_HARTREE_PER_K: float = float(_ase.kB) * EV_IN_HARTREE
#: eV/Angstrom^3 per GPa (~= 0.006242). Input pressures are quoted in GPa; ASE's stress
#: and its cell filters work in eV/Angstrom^3, so this is the conversion at that boundary.
GPA_IN_EV_PER_ANG3: float = float(_ase.GPa)

# Rydberg is defined as exactly half a Hartree; if this ever fails, ase.units changed
# in a way that invalidates the assumptions in the UPF reader.
assert abs(RYDBERG_IN_HARTREE - 0.5) < 1e-12, "Ry/Ha is not 1/2 -- check ase.units"

# --------------------------------------------------------------- string unit parsing
#: Accepted energy unit spellings mapped to their value in Hartree.
ENERGY_UNITS: dict[str, float] = {
    "ha": 1.0,
    "hartree": 1.0,
    "au": 1.0,
    "ry": RYDBERG_IN_HARTREE,
    "ryd": RYDBERG_IN_HARTREE,
    "rydberg": RYDBERG_IN_HARTREE,
    "ev": EV_IN_HARTREE,
}

#: Accepted length unit spellings mapped to their value in Bohr.
LENGTH_UNITS: dict[str, float] = {
    "bohr": 1.0,
    "a0": 1.0,
    "au": 1.0,
    "ang": ANG_IN_BOHR,
    "angstrom": ANG_IN_BOHR,
    "angstroem": ANG_IN_BOHR,
    "a": ANG_IN_BOHR,
}

_QUANTITY_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?)\s*([A-Za-z0-9_]*)\s*$"
)


def _parse_quantity(value, table: dict[str, float], default_unit: str, kind: str) -> float:
    """Parse ``value`` into the internal atomic unit using ``table``.

    ``value`` may be a number (interpreted as ``default_unit``) or a string carrying an
    explicit unit, e.g. ``"60 Ry"``, ``"816eV"``, ``"30 Ha"``. An explicit unit always
    wins over ``default_unit``; mixing up Ry and eV cutoffs is a classic and expensive
    user error, so unknown or malformed units raise rather than being guessed at.
    """
    if isinstance(value, bool):  # bool is an int subclass; never a physical quantity
        raise TypeError(f"{kind} must be a number or a string with a unit, got bool {value!r}")
    if isinstance(value, (int, float)):
        number, unit = float(value), default_unit
    elif isinstance(value, str):
        match = _QUANTITY_RE.match(value)
        if match is None:
            raise ValueError(
                f"cannot parse {kind} {value!r}; expected e.g. 60, '60 Ry', '816 eV', '30 Ha'"
            )
        number = float(match.group(1).replace("d", "e").replace("D", "e"))
        unit = match.group(2) or default_unit
    else:
        raise TypeError(
            f"{kind} must be a number or a string with a unit, got {type(value).__name__}"
        )

    factor = table.get(unit.lower())
    if factor is None:
        known = ", ".join(sorted(table))
        raise ValueError(f"unknown {kind} unit {unit!r} in {value!r}; known units: {known}")
    return number * factor


def parse_energy(value, default_unit: str = "Ry") -> float:
    """Return ``value`` in Hartree.

    The default unit is Rydberg because plane-wave cutoffs in the pslibrary/SSSP and
    Quantum ESPRESSO ecosystems -- the recommended-cutoff tables amaterasu ships against --
    are quoted in Ry. Write ``"816 eV"`` explicitly for the VASP convention.

    >>> round(parse_energy(60), 6)          # 60 Ry
    30.0
    >>> round(parse_energy("30 Ha"), 6)
    30.0
    """
    return _parse_quantity(value, ENERGY_UNITS, default_unit, "energy")


def parse_length(value, default_unit: str = "Angstrom") -> float:
    """Return ``value`` in Bohr. Defaults to Angstrom, the unit of structure files.

    >>> round(parse_length("1 Bohr"), 12)
    1.0
    """
    return _parse_quantity(value, LENGTH_UNITS, default_unit, "length")


# ------------------------------------------------------------------- output convention
#: Atomic unit of time, hbar/E_h, in seconds (~2.4189e-17 s).
ATOMIC_TIME_IN_SECONDS: float = float(_ase._hbar) / (HARTREE_IN_EV * float(_ase._e))
#: Seconds per femtosecond (exact).
FS_IN_SECONDS: float = 1e-15

#: The units every physical quantity amaterasu *reports* is expressed in.
#:
#: amaterasu computes internally in Hartree atomic units (see the module docstring) and
#: converts once, at the output boundary, so that every log line, HDF5 dataset, plot axis
#: and printed table uses the same units regardless of which internal quantity produced it.
#: Input files are a separate matter: they accept explicit units per field (see
#: :func:`parse_energy`) and default to the convention of each field's own community
#: (Ry for plane-wave cutoffs, eV for smearing, fs for MD timesteps, GPa for pressure).
OUTPUT_UNITS: dict[str, str] = {
    "energy": "eV",
    "length": "Angstrom",
    "force": "eV/Angstrom",
    "pressure": "GPa",
    "time": "s",
}


def energy_to_output(value_hartree):
    """Convert an energy from internal Hartree to the output unit (eV)."""
    return value_hartree * HARTREE_IN_EV


def length_to_output(value_bohr):
    """Convert a length from internal Bohr to the output unit (Angstrom)."""
    return value_bohr * BOHR_IN_ANG


def force_to_output(value_hartree_per_bohr):
    """Convert a force from internal Ha/Bohr to the output unit (eV/Angstrom)."""
    return value_hartree_per_bohr * HARTREE_PER_BOHR_IN_EV_PER_ANG


def pressure_to_output(value_hartree_per_bohr3):
    """Convert a pressure or stress from internal Ha/Bohr^3 to the output unit (GPa)."""
    return value_hartree_per_bohr3 * HARTREE_PER_BOHR3_IN_GPA


def time_to_output(value_atomic):
    """Convert a time from internal atomic units to the output unit (seconds).

    Times in amaterasu output are in seconds, so an MD timestep reads as ``1.0e-15 s``
    rather than ``1.0 fs``. Input files still take ``md.timestep`` in fs.
    """
    return value_atomic * ATOMIC_TIME_IN_SECONDS
