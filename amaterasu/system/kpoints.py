"""Brillouin-zone sampling: Monkhorst-Pack meshes, density-derived grids, band paths.

Conventions
-----------
k-points are stored in **fractional coordinates of the reciprocal cell**, so a point is
independent of the 2*pi convention. Where an absolute reciprocal length is needed -- the
``density`` scheme -- the 2*pi convention is used, i.e. ``b_i = 2*pi * reciprocal(a)_i``
in 1/Angstrom, matching VASP's ``KSPACING``.

k-point convergence must be established independently of the plane-wave cutoff: converge
one with the other held fixed, never both at once.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..config.schema import KpointsConfig
from .symmetry import ibz_kpoints

__all__ = ["KPointSet", "band_path", "band_path_with_cell", "generate", "grid_from_density"]


@dataclass
class KPointSet:
    """A set of k-points with integration weights summing to 1."""

    #: Fractional coordinates in the reciprocal cell, shape ``(nk, 3)``.
    points: np.ndarray
    #: Integration weights, normalised to sum to 1.
    weights: np.ndarray
    #: Mesh divisions, when the set came from a regular grid.
    mesh: tuple[int, int, int] | None
    shift: tuple[float, float, float]
    #: True if the set was reduced to the irreducible wedge.
    symmetry_reduced: bool
    #: Number of points before symmetry reduction.
    n_full: int

    def __len__(self) -> int:
        return len(self.points)

    def __str__(self) -> str:
        mesh = "x".join(str(n) for n in self.mesh) if self.mesh else "explicit"
        if self.symmetry_reduced and self.n_full != len(self):
            return f"{mesh} mesh, {self.n_full} points -> {len(self)} in the irreducible wedge"
        return f"{mesh} mesh, {len(self)} points (no symmetry reduction)"


def grid_from_density(atoms: Atoms, spacing: float) -> tuple[int, int, int]:
    """Smallest mesh whose k-point spacing is at most ``spacing`` (1/Angstrom, 2*pi convention).

    For the fcc primitive cell of Si (a = 5.43 A) the reciprocal vectors have length
    2*pi*sqrt(3)/a = 2.004 1/A, so a 0.3 1/A spacing needs ceil(2.004/0.3) = 7 divisions:

    >>> from ase.build import bulk
    >>> grid_from_density(bulk("Si", "diamond", a=5.43), 0.3)
    (7, 7, 7)
    """
    if spacing <= 0:
        raise ValueError(f"k-point spacing must be positive, got {spacing}")
    b_lengths = 2.0 * np.pi * np.linalg.norm(np.asarray(atoms.cell.reciprocal()), axis=1)
    mesh = np.maximum(1, np.ceil(b_lengths / spacing - 1e-9)).astype(int)
    return tuple(int(n) for n in mesh)


def generate(atoms: Atoms, config: KpointsConfig) -> KPointSet:
    """Build the k-point set described by ``config`` for ``atoms``."""
    shift = tuple(float(s) for s in config.shift)

    if config.scheme == "gamma":
        return KPointSet(
            points=np.zeros((1, 3)),
            weights=np.ones(1),
            mesh=(1, 1, 1),
            shift=(0.0, 0.0, 0.0),
            symmetry_reduced=False,
            n_full=1,
        )

    if config.scheme == "explicit":
        points = np.asarray(config.points, dtype=float)
        if config.weights is None:
            weights = np.full(len(points), 1.0 / len(points))
        else:
            weights = np.asarray(config.weights, dtype=float)
            total = weights.sum()
            if total <= 0:
                raise ValueError("explicit k-point weights must sum to a positive number")
            weights = weights / total
        return KPointSet(
            points=points,
            weights=weights,
            mesh=None,
            shift=shift,
            symmetry_reduced=False,
            n_full=len(points),
        )

    if config.scheme == "density":
        mesh = grid_from_density(atoms, config.density)
    else:  # monkhorst-pack
        mesh = tuple(int(n) for n in config.grid)

    points, weights = ibz_kpoints(
        atoms, mesh, shift=shift, symprec=config.symprec, use_symmetry=config.symmetry
    )
    return KPointSet(
        points=points,
        weights=weights,
        mesh=mesh,
        shift=shift,
        symmetry_reduced=config.symmetry,
        n_full=int(np.prod(mesh)),
    )


def band_path(atoms: Atoms, path: str = "auto", npoints: int = 200, symprec: float = 1e-5):
    """Return an :class:`ase.dft.kpoints.BandPath` for a band-structure calculation.

    ``path="auto"`` asks seekpath for the standardised path of Hinuma et al.,
    Comput. Mater. Sci. 128, 140 (2017). Note that seekpath defines its path in *its own*
    standardised primitive cell, which need not be the cell given here; the returned
    ``BandPath`` therefore carries the seekpath primitive cell, and a band-structure run
    must use that cell for the special-point labels to mean what they say. Use
    :func:`band_path_with_cell` to get that cell along with the path.
    """
    return band_path_with_cell(atoms, path, npoints, symprec)[0]


def band_path_with_cell(
    atoms: Atoms, path: str = "auto", npoints: int = 200, symprec: float = 1e-5
) -> tuple[object, Atoms]:
    """:func:`band_path`, plus the structure the path's labels are defined in.

    The two are returned together because they have to be used together: seekpath answers
    in its own standardised primitive cell, so a band structure computed in the *input*
    cell would carry labels that name points of a different Brillouin zone. For an explicit
    path there is no such translation and the input structure comes back unchanged.
    """
    if path != "auto":
        return atoms.cell.bandpath(path, npoints=npoints), atoms

    import seekpath

    from .symmetry import to_spglib_cell

    # seekpath calls spglib internally, which emits the same 2.7 error-handling
    # DeprecationWarning; see the note in amaterasu.system.symmetry.ibz_kpoints.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, message=".*OLD_ERROR_HANDLING.*"
        )
        result = seekpath.get_path(to_spglib_cell(atoms), with_time_reversal=True, symprec=symprec)
    segments = result["path"]
    # Collapse the list of (start, end) segments into ASE's string form, inserting a
    # comma wherever the path jumps rather than continues.
    labels: list[str] = []
    for index, (start, end) in enumerate(segments):
        if index == 0:
            labels += [start, end]
        elif start == segments[index - 1][1]:
            labels.append(end)
        else:
            labels += [",", start, end]
    ase_labels = "".join("," if x == "," else ("G" if x == "GAMMA" else x) for x in labels)

    primitive = Atoms(
        numbers=result["primitive_types"],
        scaled_positions=result["primitive_positions"],
        cell=result["primitive_lattice"],
        pbc=True,
    )
    special = {
        ("G" if name == "GAMMA" else name): np.asarray(coords)
        for name, coords in result["point_coords"].items()
    }
    return primitive.cell.bandpath(ase_labels, npoints=npoints, special_points=special), primitive
