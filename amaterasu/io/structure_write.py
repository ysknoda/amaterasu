"""Writer for the amaterasu plain-text structure format.

Round-trips everything :mod:`amaterasu.io.structure_read` understands: cell, species,
coordinates, initial magnetic moments and per-atom Cartesian constraints. Output is
always in Angstrom with fractional coordinates -- the combination that is invariant under
a uniform rescaling of the cell and therefore the least error-prone to edit by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian

__all__ = ["format_amaterasu_structure", "write_amaterasu_structure"]

_AXES = "xyz"


def _fix_masks(atoms: Atoms) -> list[str | None]:
    """Return a per-atom ``fix=`` string (or None) from the ASE constraint list."""
    masks: list[np.ndarray | None] = [None] * len(atoms)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            for index in constraint.index:
                masks[int(index)] = np.ones(3, dtype=bool)
        elif isinstance(constraint, FixCartesian):
            mask = np.asarray(constraint.mask, dtype=bool).reshape(-1)
            for index in np.atleast_1d(constraint.index):
                previous = masks[int(index)]
                masks[int(index)] = mask if previous is None else (previous | mask)
        else:
            raise ValueError(
                f"cannot represent constraint {type(constraint).__name__} in the amaterasu "
                "structure format; only FixAtoms and FixCartesian are supported"
            )
    return [
        None if m is None else "".join(a for a, on in zip(_AXES, m, strict=True) if on)
        for m in masks
    ]


def format_amaterasu_structure(atoms: Atoms, *, comment: str | None = None) -> str:
    """Render ``atoms`` as amaterasu-format text."""
    if not all(atoms.pbc):
        raise ValueError(f"amaterasu structures must be 3D-periodic, got pbc={atoms.pbc}")

    magmoms = atoms.get_initial_magnetic_moments()
    fixes = _fix_masks(atoms)

    lines: list[str] = []
    if comment:
        lines += [f"# {line}" for line in comment.splitlines()]
    lines += ["units       angstrom", "scale       1.0", "", "%block lattice_vectors"]
    lines += ["  " + "".join(f"{v:>15.10f}" for v in row) for row in np.asarray(atoms.cell)]
    lines += ["%endblock lattice_vectors", "", "%block atomic_positions fractional"]
    lines.append("#  el" + "".join(f"{axis:>15}" for axis in _AXES))

    for symbol, scaled, magmom, fix in zip(
        atoms.get_chemical_symbols(), atoms.get_scaled_positions(), magmoms, fixes, strict=True
    ):
        row = f"  {symbol:<3}" + "".join(f"{v:>15.10f}" for v in scaled)
        if magmom != 0.0:
            row += f"   magmom={magmom:g}"
        if fix:
            row += f"   fix={fix}"
        lines.append(row)

    lines += ["%endblock atomic_positions", ""]
    return "\n".join(lines)


def write_amaterasu_structure(
    atoms: Atoms, path: str | os.PathLike, *, comment: str | None = None
) -> None:
    """Write ``atoms`` to ``path`` in the amaterasu structure format."""
    Path(path).write_text(format_amaterasu_structure(atoms, comment=comment), encoding="utf-8")
