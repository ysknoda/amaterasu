"""Structure inspection and summary.

amaterasu carries structures as plain :class:`ase.Atoms` rather than wrapping them in a
new class: ASE is a hard dependency, every optimiser, MD integrator and NEB driver speaks
``Atoms``, and a parallel structure type would only create conversion points where the two
could drift apart. What lives here is the analysis that ``Atoms`` does not provide.

:func:`summarize` is what ``amaterasu info`` prints. Its job is to make an input mistake
visible immediately -- a transposed lattice matrix, Bohr read as Angstrom, or fractional
coordinates read as Cartesian all produce a structure that parses cleanly but has an
obviously wrong volume, density or nearest-neighbour distance.

All reported quantities follow the output convention in :mod:`amaterasu.units`:
lengths in Angstrom, energies in eV, forces in eV/Angstrom, pressures in GPa, times in s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian
from ase.units import _amu

from .symmetry import SymmetryInfo, analyze, symprec_scan

__all__ = ["StructureSummary", "minimum_distances", "summarize"]

#: Below this separation two atoms are almost certainly a unit or coordinate-type mistake
#: rather than a real structure: the shortest bond in nature (H2) is 0.74 Angstrom.
_IMPLAUSIBLE_DISTANCE_ANG = 0.5
#: Number of grams in one atomic mass unit.
_AMU_IN_GRAMS = _amu * 1e3


@dataclass
class StructureSummary:
    """Everything ``amaterasu info`` reports about a structure. Lengths in Angstrom."""

    formula: str
    n_atoms: int
    cell: np.ndarray
    lengths: np.ndarray
    angles: np.ndarray
    volume: float
    density: float
    symmetry: SymmetryInfo
    nearest_distance: float
    nearest_pair: tuple[int, int]
    species_counts: dict[str, int]
    magmoms: np.ndarray | None
    fixed_atoms: int
    warnings: list[str]

    def render(self, *, symprec_sensitivity: list[tuple[float, int, str]] | None = None) -> str:
        """Format the summary as the block of text printed by ``amaterasu info``."""
        a, b, c = self.lengths
        alpha, beta, gamma = self.angles
        species = "  ".join(f"{s}:{n}" for s, n in sorted(self.species_counts.items()))
        lines = [
            f"formula              {self.formula}   ({self.n_atoms} atoms;  {species})",
            "lattice vectors [A]",
            *(
                f"  a{i + 1}  {row[0]:14.8f} {row[1]:14.8f} {row[2]:14.8f}"
                for i, row in enumerate(self.cell)
            ),
            f"lengths     [A]      a = {a:.6f}   b = {b:.6f}   c = {c:.6f}",
            f"angles    [deg]      alpha = {alpha:.4f}   beta = {beta:.4f}   gamma = {gamma:.4f}",
            f"volume    [A^3]      {self.volume:.6f}",
            f"density [g/cm^3]     {self.density:.6f}",
            f"space group          {self.symmetry}",
            f"inequivalent atoms   {self.symmetry.n_inequivalent_atoms}"
            f"   (Wyckoff: {' '.join(self.symmetry.wyckoffs)})",
            f"nearest neighbour    {self.nearest_distance:.6f} A"
            f"   (atoms {self.nearest_pair[0]}-{self.nearest_pair[1]})",
        ]
        if self.magmoms is not None:
            total = float(np.sum(self.magmoms))
            per_atom = np.array2string(self.magmoms, precision=3)
            lines.append(f"initial magmoms      total {total:+.4f} mu_B   {per_atom}")
        if self.fixed_atoms:
            lines.append(f"constrained atoms    {self.fixed_atoms}")
        if symprec_sensitivity is not None:
            rendered = "  ".join(f"{t:g}->{num}" for t, num, _ in symprec_sensitivity)
            lines.append(f"symprec sensitivity  {rendered}")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def minimum_distances(atoms: Atoms) -> tuple[float, tuple[int, int]]:
    """Return the shortest interatomic distance under the minimum-image convention.

    For a single atom in the cell the nearest neighbour is its own periodic image, so the
    shortest lattice vector is used instead.
    """
    if len(atoms) == 1:
        # The only neighbours are periodic images; the closest sits one lattice vector away.
        lengths = np.linalg.norm(np.asarray(atoms.cell[:]), axis=1)
        return float(lengths.min()), (0, 0)
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    flat = int(np.argmin(distances))
    i, j = divmod(flat, len(atoms))
    return float(distances[i, j]), (i, j)


def _count_fixed(atoms: Atoms) -> int:
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed.update(int(k) for k in constraint.index)
        elif isinstance(constraint, FixCartesian):
            fixed.update(int(k) for k in np.atleast_1d(constraint.index))
    return len(fixed)


def summarize(atoms: Atoms, *, symprec: float = 1e-5) -> StructureSummary:
    """Analyse ``atoms`` and collect everything ``amaterasu info`` reports."""
    cell = np.asarray(atoms.cell[:], dtype=float)
    volume = float(abs(np.linalg.det(cell)))
    mass_grams = float(atoms.get_masses().sum()) * _AMU_IN_GRAMS
    density = mass_grams / (volume * 1e-24) if volume > 0 else float("nan")

    nearest, pair = minimum_distances(atoms)
    magmoms = atoms.get_initial_magnetic_moments()
    symbols = atoms.get_chemical_symbols()

    warnings: list[str] = []
    if nearest < _IMPLAUSIBLE_DISTANCE_ANG:
        warnings.append(
            f"nearest-neighbour distance is {nearest:.4f} A, shorter than any real bond "
            f"({_IMPLAUSIBLE_DISTANCE_ANG} A). Check the length unit and whether the "
            "coordinates are fractional or Cartesian."
        )
    if np.linalg.det(cell) < 0:
        warnings.append(
            "the lattice vectors form a left-handed set (det < 0); some conventions and "
            "external tools assume a right-handed cell."
        )

    return StructureSummary(
        formula=atoms.get_chemical_formula(),
        n_atoms=len(atoms),
        cell=cell,
        lengths=np.asarray(atoms.cell.lengths(), dtype=float),
        angles=np.asarray(atoms.cell.angles(), dtype=float),
        volume=volume,
        density=density,
        symmetry=analyze(atoms, symprec=symprec),
        nearest_distance=nearest,
        nearest_pair=pair,
        species_counts={s: symbols.count(s) for s in dict.fromkeys(symbols)},
        magmoms=magmoms if np.any(magmoms != 0.0) else None,
        fixed_atoms=_count_fixed(atoms),
        warnings=warnings,
    )


def summarize_with_sensitivity(atoms: Atoms, *, symprec: float = 1e-5) -> str:
    """Rendered summary including the symprec sensitivity scan."""
    return summarize(atoms, symprec=symprec).render(symprec_sensitivity=symprec_scan(atoms))
