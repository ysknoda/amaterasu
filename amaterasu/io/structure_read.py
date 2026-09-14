"""Reader for the amaterasu plain-text structure format (and ASE formats).

Format
------
Free-form, line oriented. ``#`` starts a comment that runs to the end of the line.
Blank lines are ignored. Keywords and block names are case-insensitive; chemical symbols
are case-sensitive. The ``%block``/``%endblock`` convention follows SIESTA fdf and
CASTEP ``.cell``::

    units       angstrom          # angstrom (default) | bohr
    scale       1.0               # uniform multiplier applied to the lattice vectors

    %block lattice_vectors        # rows are a1, a2, a3
      0.00000   2.71535   2.71535
      2.71535   0.00000   2.71535
      2.71535   2.71535   0.00000
    %endblock lattice_vectors

    %block atomic_positions fractional     # fractional | cartesian -- REQUIRED
    #  el      x        y        z      [magmom=...] [fix=...]
       Si    0.0000   0.0000   0.0000
       Si    0.2500   0.2500   0.2500   magmom=0.0  fix=z
    %endblock atomic_positions

``units`` applies to the lattice vectors and to *cartesian* atomic positions; fractional
coordinates are dimensionless. The ``fractional``/``cartesian`` modifier is mandatory
rather than defaulted: silently misreading one as the other produces a structure that is
wrong but superficially plausible, which is far more expensive than an up-front error.

Per-atom attributes
    ``magmom=<float>``  initial collinear magnetic moment in Bohr magnetons.
    ``fix=<axes>``      subset of ``xyz`` held fixed during relaxation/NEB/MD; mapped to
                        :class:`ase.constraints.FixCartesian` (where ``True`` == fixed).

Errors are accumulated and reported together with line numbers, so a malformed file
surfaces every problem in one pass instead of one error per edit-rerun cycle.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixCartesian
from ase.data import chemical_symbols

from ..units import BOHR_IN_ANG

__all__ = [
    "StructureParseError",
    "looks_like_amaterasu_format",
    "read_amaterasu_structure",
    "read_structure",
]

_LENGTH_UNIT_TO_ANG = {
    "angstrom": 1.0,
    "angstroem": 1.0,
    "ang": 1.0,
    "a": 1.0,
    "bohr": BOHR_IN_ANG,
    "au": BOHR_IN_ANG,
    "a0": BOHR_IN_ANG,
}
_FRACTIONAL_WORDS = frozenset({"fractional", "frac", "crystal", "direct", "reduced"})
_CARTESIAN_WORDS = frozenset({"cartesian", "cart", "absolute"})
_KNOWN_SYMBOLS = frozenset(chemical_symbols[1:])
_MIN_CELL_VOLUME_ANG3 = 1e-8


class StructureParseError(ValueError):
    """Raised when a structure file cannot be parsed. Carries every problem found."""

    def __init__(self, source: str, problems: list[tuple[int | None, str]]):
        self.source = source
        # Sorted by line number so the report reads in file order; file-level problems
        # (lineno None) come first.
        self.problems = sorted(problems, key=lambda p: (p[0] is not None, p[0] or 0))
        lines = [f"{len(self.problems)} problem(s) in {source}:"]
        for lineno, msg in self.problems:
            prefix = f"  line {lineno}: " if lineno is not None else "  "
            body = msg.splitlines()
            lines.append(prefix + body[0])
            lines.extend(" " * len(prefix) + cont for cont in body[1:])
        super().__init__("\n".join(lines))


class _ErrorLog:
    """Accumulates parse problems so they can all be reported in one exception."""

    def __init__(self, source: str):
        self.source = source
        self.problems: list[tuple[int | None, str]] = []

    def add(self, lineno: int | None, message: str) -> None:
        self.problems.append((lineno, message))

    def raise_if_any(self) -> None:
        if self.problems:
            raise StructureParseError(self.source, self.problems)


def looks_like_amaterasu_format(text: str) -> bool:
    """True if ``text`` contains a ``%block`` directive, i.e. is the amaterasu format."""
    return any(line.lstrip().lower().startswith("%block") for line in text.splitlines())


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def _tokenize(text: str) -> list[tuple[int, str]]:
    """Return ``(1-based line number, content)`` for every non-blank, non-comment line."""
    out = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        content = _strip_comment(raw).strip()
        if content:
            out.append((lineno, content))
    return out


def _parse_float(token: str) -> float:
    """Parse a float, accepting Fortran-style ``d`` exponents (``1.0d-3``)."""
    return float(token.replace("d", "e").replace("D", "E"))


def _split_blocks(lines, log):
    """Separate ``%block``/``%endblock`` regions from standalone ``key value`` lines.

    Returns ``(keywords, blocks)`` where ``keywords`` maps a lowercased key to
    ``(lineno, value_string)`` and ``blocks`` maps a lowercased block name to
    ``(header_lineno, modifiers, body_lines)``.
    """
    keywords: dict[str, tuple[int, str]] = {}
    blocks: dict[str, tuple[int, list[str], list[tuple[int, str]]]] = {}
    current: str | None = None
    header_lineno = 0
    modifiers: list[str] = []
    body: list[tuple[int, str]] = []

    for lineno, content in lines:
        lowered = content.lower()
        if lowered.startswith("%block"):
            if current is not None:
                log.add(
                    lineno,
                    f"'%block' inside block '{current}' unterminated since line {header_lineno}",
                )
                blocks.setdefault(current, (header_lineno, modifiers, body))
            parts = content.split()[1:]
            if not parts:
                log.add(lineno, "'%block' is missing a block name")
                current, modifiers = "<unnamed>", []
            else:
                current, modifiers = parts[0].lower(), [p.lower() for p in parts[1:]]
            if current in blocks:
                log.add(
                    lineno,
                    f"duplicate block '{current}' (first defined at line {blocks[current][0]})",
                )
            header_lineno, body = lineno, []
        elif lowered.startswith("%endblock"):
            if current is None:
                log.add(lineno, "'%endblock' without a matching '%block'")
                continue
            parts = content.split()[1:]
            if parts and parts[0].lower() != current:
                log.add(
                    lineno,
                    f"'%endblock {parts[0]}' does not match '%block {current}'"
                    f" at line {header_lineno}",
                )
            blocks[current] = (header_lineno, modifiers, body)
            current = None
        elif current is not None:
            body.append((lineno, content))
        else:
            parts = content.split(None, 1)
            key = parts[0].lower()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key in keywords:
                log.add(lineno, f"duplicate keyword '{key}' (first set at line {keywords[key][0]})")
            keywords[key] = (lineno, value)

    if current is not None:
        log.add(header_lineno, f"block '{current}' is never closed with '%endblock'")
        blocks.setdefault(current, (header_lineno, modifiers, body))
    return keywords, blocks


def _parse_scalar_keywords(keywords, log):
    """Extract ``units`` and ``scale``, returning ``(length_unit_in_ang, scale)``."""
    unit_factor = 1.0
    if "units" in keywords:
        lineno, value = keywords["units"]
        factor = _LENGTH_UNIT_TO_ANG.get(value.strip().lower())
        if factor is None:
            log.add(
                lineno,
                f"unknown length unit {value!r}; known: {', '.join(sorted(_LENGTH_UNIT_TO_ANG))}",
            )
        else:
            unit_factor = factor

    scale = 1.0
    if "scale" in keywords:
        lineno, value = keywords["scale"]
        try:
            scale = _parse_float(value.split()[0])
        except (ValueError, IndexError):
            log.add(lineno, f"'scale' expects one number, got {value!r}")
        else:
            if scale <= 0.0:
                log.add(lineno, f"'scale' must be positive, got {scale}")
                scale = 1.0

    for key, (lineno, _) in keywords.items():
        if key not in ("units", "scale"):
            log.add(lineno, f"unknown keyword {key!r}; expected 'units' or 'scale'")
    return unit_factor, scale


def _parse_lattice(blocks, log, unit_factor, scale):
    """Parse ``%block lattice_vectors`` into a 3x3 cell in Angstrom."""
    if "lattice_vectors" not in blocks:
        log.add(None, "required block 'lattice_vectors' is missing")
        return None
    header_lineno, modifiers, body = blocks["lattice_vectors"]
    if modifiers:
        log.add(header_lineno, f"'lattice_vectors' takes no modifiers, got {' '.join(modifiers)}")

    rows = []
    for lineno, content in body:
        tokens = content.split()
        if len(tokens) != 3:
            log.add(
                lineno, f"lattice vector needs exactly 3 numbers, got {len(tokens)}: {content!r}"
            )
            continue
        try:
            rows.append([_parse_float(t) for t in tokens])
        except ValueError as exc:
            log.add(lineno, f"cannot parse lattice vector {content!r}: {exc}")

    if len(rows) != 3:
        log.add(
            header_lineno, f"'lattice_vectors' needs exactly 3 rows (a1, a2, a3), got {len(rows)}"
        )
        return None

    cell = np.array(rows, dtype=float) * unit_factor * scale
    volume = float(np.linalg.det(cell))
    if abs(volume) < _MIN_CELL_VOLUME_ANG3:
        log.add(
            header_lineno, f"lattice vectors are singular or degenerate (volume = {volume:.3e} A^3)"
        )
        return None
    return cell


def _parse_atom_attributes(tokens, lineno, log):
    """Parse trailing ``key=value`` attributes on an atomic-position line."""
    magmom, fix_mask = 0.0, None
    for token in tokens:
        if "=" not in token:
            log.add(
                lineno,
                f"unexpected token {token!r}; per-atom attributes look like"
                " 'magmom=1.0' or 'fix=z'",
            )
            continue
        key, _, value = token.partition("=")
        key = key.lower()
        if key == "magmom":
            try:
                magmom = _parse_float(value)
            except ValueError:
                log.add(lineno, f"'magmom=' expects a number, got {value!r}")
        elif key == "fix":
            axes = value.lower()
            if not axes or set(axes) - set("xyz"):
                log.add(
                    lineno, f"'fix=' expects a subset of 'xyz' (e.g. fix=z, fix=xyz), got {value!r}"
                )
            else:
                fix_mask = tuple(axis in axes for axis in "xyz")
        else:
            log.add(lineno, f"unknown per-atom attribute {key!r}; known: 'magmom', 'fix'")
    return magmom, fix_mask


def _parse_positions(blocks, log, unit_factor):
    """Parse ``%block atomic_positions`` into symbols, coordinates, magmoms and fix masks."""
    if "atomic_positions" not in blocks:
        log.add(None, "required block 'atomic_positions' is missing")
        return None
    header_lineno, modifiers, body = blocks["atomic_positions"]

    fractional = None
    for modifier in modifiers:
        if modifier in _FRACTIONAL_WORDS:
            fractional = True
        elif modifier in _CARTESIAN_WORDS:
            fractional = False
        else:
            log.add(
                header_lineno,
                f"unknown modifier {modifier!r} on 'atomic_positions';"
                " use 'fractional' or 'cartesian'",
            )
    if fractional is None:
        log.add(
            header_lineno,
            "'atomic_positions' requires a coordinate type: "
            "'%block atomic_positions fractional' or '... cartesian'. "
            "This is never defaulted -- silently confusing the two yields a structure "
            "that is wrong but superficially plausible.",
        )
        # Keep going anyway: the atom lines are still worth validating so the user sees
        # every problem in one pass. Only the final return is suppressed.

    symbols, coords, magmoms, fix_masks = [], [], [], []
    for lineno, content in body:
        tokens = content.split()
        if len(tokens) < 4:
            log.add(
                lineno, f"atom line needs 'symbol x y z', got {len(tokens)} token(s): {content!r}"
            )
            continue
        symbol = tokens[0]
        if symbol not in _KNOWN_SYMBOLS:
            log.add(
                lineno,
                f"unknown chemical symbol {symbol!r} (case-sensitive: 'Si', not 'SI' or 'si')",
            )
            continue
        try:
            position = [_parse_float(t) for t in tokens[1:4]]
        except ValueError as exc:
            log.add(lineno, f"cannot parse coordinates in {content!r}: {exc}")
            continue
        magmom, fix_mask = _parse_atom_attributes(tokens[4:], lineno, log)
        symbols.append(symbol)
        coords.append(position)
        magmoms.append(magmom)
        fix_masks.append(fix_mask)

    if not symbols:
        log.add(header_lineno, "'atomic_positions' contains no valid atoms")
        return None

    if fractional is None:
        return None
    coords = np.array(coords, dtype=float)
    if not fractional:
        coords = coords * unit_factor
    return symbols, coords, np.array(magmoms), fix_masks, fractional


def _build_constraints(fix_masks):
    """Group per-atom fix masks into as few FixCartesian constraints as possible."""
    grouped: dict[tuple[bool, bool, bool], list[int]] = {}
    for index, mask in enumerate(fix_masks):
        if mask is not None:
            grouped.setdefault(mask, []).append(index)
    return [FixCartesian(indices, mask=mask) for mask, indices in sorted(grouped.items())]


def read_amaterasu_structure(source: str | os.PathLike, *, text: str | None = None) -> Atoms:
    """Parse the amaterasu structure format and return an :class:`ase.Atoms`.

    Parameters
    ----------
    source
        Path to the file, or a label used in error messages when ``text`` is given.
    text
        Optional file content; when omitted, ``source`` is read from disk.

    Raises
    ------
    StructureParseError
        With *every* problem found in the file, each tagged with its line number.
    """
    label = str(source)
    if text is None:
        text = Path(source).read_text(encoding="utf-8")

    log = _ErrorLog(label)
    keywords, blocks = _split_blocks(_tokenize(text), log)

    for name, (lineno, _, _) in blocks.items():
        if name not in ("lattice_vectors", "atomic_positions"):
            log.add(
                lineno, f"unknown block {name!r}; expected 'lattice_vectors' or 'atomic_positions'"
            )

    unit_factor, scale = _parse_scalar_keywords(keywords, log)
    cell = _parse_lattice(blocks, log, unit_factor, scale)
    parsed = _parse_positions(blocks, log, unit_factor)
    log.raise_if_any()

    symbols, coords, magmoms, fix_masks, fractional = parsed
    atoms = Atoms(symbols=symbols, cell=cell, pbc=True)
    if fractional:
        atoms.set_scaled_positions(coords)
    else:
        atoms.set_positions(coords)
    if np.any(magmoms != 0.0):
        atoms.set_initial_magnetic_moments(magmoms)
    constraints = _build_constraints(fix_masks)
    if constraints:
        atoms.set_constraint(constraints)
    return atoms


def read_structure(path: str | os.PathLike, *, format: str | None = None) -> Atoms:
    """Read a structure from ``path``, auto-detecting the amaterasu format.

    Any file containing a ``%block`` directive is parsed as the amaterasu format;
    everything else is handed to :func:`ase.io.read`, which covers POSCAR/CONTCAR, CIF,
    xyz/extxyz and the rest of the ASE format zoo. Pass ``format`` to force one of
    ``"amaterasu"`` or any ASE format name.
    """
    path = Path(path)
    if format is None:
        text = path.read_text(encoding="utf-8", errors="replace")
        format = "amaterasu" if looks_like_amaterasu_format(text) else None
        if format == "amaterasu":
            return read_amaterasu_structure(path, text=text)
    elif format == "amaterasu":
        return read_amaterasu_structure(path)

    from ase.io import read as ase_read

    atoms = ase_read(path, format=format)
    if not all(atoms.pbc):
        # amaterasu is a 3D-periodic code; a non-periodic file is almost certainly a
        # molecule the user meant to put in a box, so say so rather than inventing one.
        raise ValueError(
            f"{path} has pbc={atoms.pbc}; amaterasu requires a fully 3D-periodic cell. "
            "Add explicit lattice vectors (e.g. put the molecule in a vacuum box)."
        )
    return atoms
