"""amaterasu -- plane-wave density functional theory for 3D periodic crystals.

amaterasu computes internally in Hartree atomic units and reports every physical quantity
in eV, Angstrom, eV/Angstrom, eV/Angstrom^3 (stress, positive under compression), GPa
and fs; see :mod:`amaterasu.units`.
"""

from __future__ import annotations

__all__ = ["__version__"]

try:  # pragma: no cover - trivial packaging fallback
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("amaterasu")
except (ImportError, PackageNotFoundError):  # pragma: no cover
    __version__ = "0.1.0a2"
