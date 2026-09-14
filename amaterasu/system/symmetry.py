"""Crystal symmetry analysis, via spglib.

Everything symmetry-related goes through this module so that a single ``symprec`` is
applied consistently: the space group reported in the log, the irreducible k-point set
used for the Brillouin-zone sum, and (from Phase 3 on) the symmetrisation of the density,
forces and stress must all agree, or the calculation is quietly inconsistent.

``symprec`` is a real physical choice, not a formality. A structure that has been relaxed
with finite forces sits slightly off its ideal Wyckoff positions, and a too-tight
tolerance will report P1 for what is physically a cubic crystal -- collapsing the
irreducible wedge to the full zone and multiplying the cost. :func:`symprec_scan` exists
so that sensitivity can be checked rather than assumed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms

__all__ = [
    "SymmetryInfo",
    "analyze",
    "atom_mapping",
    "cartesian_operations",
    "ibz_kpoints",
    "spin_decorated",
    "symmetrize_sphere",
    "symmetrize_tensor",
    "symmetrize_vectors",
    "symprec_scan",
    "to_spglib_cell",
]

#: Tolerances used by :func:`symprec_scan`, spanning "tight" to "generous".
_SCAN_TOLERANCES = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


@dataclass(frozen=True)
class SymmetryInfo:
    """Space-group information for a structure at one ``symprec``."""

    number: int
    international: str
    pointgroup: str
    hall_number: int
    #: Rotation matrices in fractional coordinates, shape ``(n_ops, 3, 3)``.
    rotations: np.ndarray
    #: Fractional translations, shape ``(n_ops, 3)``.
    translations: np.ndarray
    #: Wyckoff letter per atom.
    wyckoffs: tuple[str, ...]
    #: Index of the first symmetry-equivalent atom, per atom.
    equivalent_atoms: np.ndarray
    symprec: float

    @property
    def n_operations(self) -> int:
        return len(self.rotations)

    @property
    def n_inequivalent_atoms(self) -> int:
        return len(set(self.equivalent_atoms.tolist()))

    def __str__(self) -> str:
        return (
            f"{self.international} ({self.number}), point group {self.pointgroup}, "
            f"{self.n_operations} operations (symprec={self.symprec:g})"
        )


def to_spglib_cell(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an :class:`ase.Atoms` to the ``(lattice, positions, numbers)`` spglib wants."""
    return (
        np.asarray(atoms.cell[:], dtype=float),
        np.asarray(atoms.get_scaled_positions(), dtype=float),
        np.asarray(atoms.numbers, dtype=np.intc),
    )


def analyze(atoms: Atoms, symprec: float = 1e-5) -> SymmetryInfo:
    """Determine the space group of ``atoms``.

    Raises
    ------
    RuntimeError
        If spglib cannot identify a space group, which usually means overlapping atoms
        or a degenerate cell.
    """
    # _throw=True selects spglib's current error handling; the legacy None-return path is
    # deprecated and emits a warning.
    try:
        dataset = spglib.get_symmetry_dataset(to_spglib_cell(atoms), symprec=symprec, _throw=True)
    except Exception as exc:  # spglib raises its own C++-backed exception type
        raise RuntimeError(
            f"spglib could not determine the symmetry at symprec={symprec:g} ({exc}); "
            "check for overlapping atoms or a singular cell"
        ) from exc
    if dataset is None:  # pragma: no cover - defensive; _throw should have raised
        raise RuntimeError(f"spglib returned no dataset at symprec={symprec:g}")
    return SymmetryInfo(
        number=int(dataset.number),
        international=str(dataset.international),
        pointgroup=str(dataset.pointgroup),
        hall_number=int(dataset.hall_number),
        rotations=np.asarray(dataset.rotations),
        translations=np.asarray(dataset.translations),
        wyckoffs=tuple(dataset.wyckoffs),
        equivalent_atoms=np.asarray(dataset.equivalent_atoms),
        symprec=symprec,
    )


def symprec_scan(atoms: Atoms, tolerances=_SCAN_TOLERANCES) -> list[tuple[float, int, str]]:
    """Return ``(symprec, space-group number, symbol)`` across a range of tolerances.

    A structure whose space group changes across this range is sitting near a symmetry
    boundary; the chosen ``symprec`` then materially affects the k-point count and any
    symmetrisation, and should be stated explicitly rather than left at the default.
    """
    results = []
    for tol in tolerances:
        try:
            info = analyze(atoms, symprec=tol)
        except RuntimeError:
            results.append((tol, 0, "unresolved"))
        else:
            results.append((tol, info.number, info.international))
    return results


def ibz_kpoints(
    atoms: Atoms,
    mesh: tuple[int, int, int],
    *,
    shift=(0.0, 0.0, 0.0),
    symprec: float = 1e-5,
    time_reversal: bool = True,
    use_symmetry: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a Monkhorst-Pack mesh to the irreducible Brillouin zone.

    Parameters
    ----------
    mesh
        Number of divisions along each reciprocal lattice vector.
    shift
        Half-grid shift per direction; ``0.5`` shifts by half a mesh step. Note that
        spglib takes this as 0 or 1 meaning "no shift" / "half-step shift", so any
        non-zero entry here is passed as a half-step.
    use_symmetry
        When False the full mesh is returned with uniform weights, which is what a
        calculation with a symmetry-breaking perturbation (a finite field, a displaced
        atom) requires.

    Returns
    -------
    kpoints
        Fractional coordinates in the reciprocal cell, shape ``(n_irreducible, 3)``.
    weights
        Normalised weights summing to 1.
    """
    mesh = np.asarray(mesh, dtype=np.intc)
    if mesh.shape != (3,) or np.any(mesh < 1):
        raise ValueError(f"mesh must be three positive integers, got {mesh.tolist()}")
    is_shift = np.asarray([1 if abs(s) > 1e-12 else 0 for s in shift], dtype=np.intc)

    if not use_symmetry:
        grid = np.stack(np.meshgrid(*[np.arange(n) for n in mesh], indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        kpoints = (grid + 0.5 * is_shift) / mesh
        kpoints -= np.rint(kpoints)  # fold into (-1/2, 1/2]
        weights = np.full(len(kpoints), 1.0 / len(kpoints))
        return kpoints, weights

    # spglib 2.7 emits a DeprecationWarning from inside get_ir_reciprocal_mesh about its
    # error-handling transition. The documented opt-out is a module-level global
    # (spglib.error.OLD_ERROR_HANDLING), which a library has no business setting on a
    # caller's behalf, and this function accepts no per-call _throw. So the warning is
    # silenced only around this call, and the result is validated directly below instead.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, message=".*OLD_ERROR_HANDLING.*"
        )
        mapping, grid = spglib.get_ir_reciprocal_mesh(
            mesh,
            to_spglib_cell(atoms),
            is_shift=is_shift,
            is_time_reversal=time_reversal,
            symprec=symprec,
        )
    expected = int(np.prod(mesh))
    if mapping is None or len(mapping) != expected:
        raise RuntimeError(
            f"spglib returned {0 if mapping is None else len(mapping)} grid points for a "
            f"{mesh.tolist()} mesh (expected {expected}) at symprec={symprec:g}"
        )
    unique, counts = np.unique(mapping, return_counts=True)
    kpoints = (grid[unique] + 0.5 * is_shift) / mesh
    kpoints -= np.rint(kpoints)
    weights = counts / counts.sum()
    return kpoints, weights


def spin_decorated(atoms: Atoms, magmoms=None) -> Atoms:
    """A copy of ``atoms`` in which sites with different initial moments are distinct species.

    spglib sees only atomic numbers, so it would call the two iron sublattices of an
    antiferromagnet equivalent and hand back an irreducible wedge -- and a density
    symmetrisation -- that averages the two moments to zero. Giving inequivalent moments
    inequivalent (fictitious) atomic numbers restores the magnetic symmetry for the
    collinear case, which is the standard treatment.

    Sites are grouped by ``(symbol, moment)`` with the moment rounded to 1e-4 muB, so
    numerically identical moments are not split apart by rounding noise.
    """
    if magmoms is None:
        magmoms = atoms.get_initial_magnetic_moments()
    magmoms = np.asarray(magmoms, dtype=float).reshape(len(atoms), -1)[:, 0]
    if np.allclose(magmoms, magmoms[0] if len(magmoms) else 0.0):
        return atoms

    decorated = atoms.copy()
    keys = {}
    numbers = []
    for number, moment in zip(atoms.numbers, np.round(magmoms, 4), strict=True):
        key = (int(number), float(moment))
        # Offsets of 1000 keep the fictitious numbers clear of every real element.
        numbers.append(keys.setdefault(key, int(number) + 1000 * len(keys)))
    decorated.numbers = np.asarray(numbers)
    return decorated


def cartesian_operations(symmetry: SymmetryInfo, cell: np.ndarray) -> np.ndarray:
    """Point operations as row-vector Cartesian matrices: ``r' = r @ M``.

    spglib works in fractional coordinates with a column convention,
    ``x' = R x + f``. With lattice vectors as the *rows* of ``cell`` a Cartesian row
    vector is ``r = x_row @ cell``, so ``M = cell^-1 R^T cell``. For a genuine symmetry
    operation ``M`` comes out orthogonal, which is what lets forces be rotated with the
    same matrix as positions.
    """
    inverse = np.linalg.inv(np.asarray(cell, dtype=float))
    return np.einsum(
        "ij,sjk,kl->sil",
        inverse,
        np.asarray(symmetry.rotations, dtype=float).transpose(0, 2, 1),
        cell,
    )


def atom_mapping(atoms: Atoms, symmetry: SymmetryInfo) -> np.ndarray:
    """``mapping[s, a]`` is the atom that operation ``s`` sends atom ``a`` to.

    Raises
    ------
    ValueError
        If an operation does not map the structure onto itself within ``symprec``, which
        means the symmetry data and the structure have drifted apart.
    """
    fractional = np.asarray(atoms.get_scaled_positions(), dtype=float)
    numbers = np.asarray(atoms.numbers)
    tolerance = max(10.0 * symmetry.symprec, 1e-5)
    mapping = np.empty((symmetry.n_operations, len(atoms)), dtype=int)
    for index, (rotation, translation) in enumerate(
        zip(symmetry.rotations, symmetry.translations, strict=True)
    ):
        images = fractional @ np.asarray(rotation, dtype=float).T + translation
        difference = images[:, None, :] - fractional[None, :, :]
        difference -= np.rint(difference)
        distance = np.linalg.norm(difference, axis=-1)
        distance[numbers[:, None] != numbers[None, :]] = np.inf
        target = np.argmin(distance, axis=1)
        if np.max(distance[np.arange(len(atoms)), target]) > tolerance:
            raise ValueError(
                f"symmetry operation {index} does not map the structure onto itself "
                f"within {tolerance:g} (fractional); the structure and the symmetry data "
                "are inconsistent"
            )
        mapping[index] = target
    return mapping


def symmetrize_vectors(
    vectors: np.ndarray, atoms: Atoms, symmetry: SymmetryInfo, cell: np.ndarray
) -> np.ndarray:
    """Average a per-atom Cartesian vector field over the space group.

    A Brillouin-zone sum restricted to the irreducible wedge gives the right *scalar* but
    not the right *vector*: the wedge carries only part of the point group, so the
    accumulated vector has the symmetry of the wedge instead of the crystal. Averaging
    ``F_a = (1/N) sum_S F_{S(a)} M_S^-1`` restores it, and makes a force on a
    high-symmetry site come out exactly zero rather than merely small.
    """
    vectors = np.asarray(vectors, dtype=float)
    if symmetry.n_operations <= 1:
        return vectors.copy()
    operations = cartesian_operations(symmetry, cell)
    mapping = atom_mapping(atoms, symmetry)
    averaged = np.zeros_like(vectors)
    for index, matrix in enumerate(operations):
        averaged += vectors[mapping[index]] @ np.linalg.inv(matrix)
    return averaged / symmetry.n_operations


def symmetrize_tensor(tensor: np.ndarray, symmetry: SymmetryInfo, cell: np.ndarray) -> np.ndarray:
    """Average a rank-2 Cartesian tensor (the stress) over the point group.

    The same argument as for vectors applies, with one useful consequence: a cubic
    crystal's symmetrised stress is exactly isotropic, so any off-diagonal residue in the
    raw tensor is a direct measure of the Brillouin-zone and grid discretisation.
    """
    tensor = np.asarray(tensor, dtype=float)
    if symmetry.n_operations <= 1:
        return tensor.copy()
    operations = cartesian_operations(symmetry, cell)
    averaged = np.zeros_like(tensor)
    for matrix in operations:
        averaged += matrix.T @ tensor @ matrix
    return averaged / symmetry.n_operations


def symmetrize_sphere(
    coefficients: np.ndarray,
    sphere,
    grid,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    """Symmetrise Fourier coefficients of a scalar field over the space group.

    A field invariant under ``{R|f}`` -- fractional ``r -> R r + f`` -- satisfies

        n_sym(G) = (1/N) sum_S n(R_S^T G) exp(-2 pi i G.f_S)

    with ``G`` the integer Miller indices. This is what makes an irreducible-wedge k-point
    sum legitimate: the density accumulated from the wedge alone has the symmetry of the
    wedge, not of the crystal, and only becomes the true density after this average.

    The map ``G -> R^T G`` keeps a vector inside the cutoff sphere. In Cartesian terms it
    is the transposed point operation ``W^T``, which is orthogonal, so ``|G|`` is
    unchanged and no coefficient can be rotated out of the basis.
    """
    coefficients = np.asarray(coefficients)
    rotations = np.asarray(rotations, dtype=int)
    translations = np.asarray(translations, dtype=float)
    if len(rotations) <= 1:
        return coefficients.copy()

    # Grid position of every Miller index in the sphere, for O(1) lookup of the image.
    lookup = np.full(grid.shape, -1, dtype=np.int64)
    lookup[grid.indices(sphere)] = np.arange(len(sphere))

    miller = sphere.miller
    accumulated = np.zeros_like(coefficients)
    for rotation, translation in zip(rotations, translations, strict=True):
        rotated = miller @ rotation  # (R^T G)_i = R_ji G_j, i.e. G @ R
        image = lookup[tuple(rotated[:, axis] % grid.shape[axis] for axis in range(3))]
        if np.any(image < 0):
            raise ValueError(
                "a symmetry operation maps a G-vector outside the cutoff sphere; "
                "the cell and the symmetry data are inconsistent"
            )
        phase = np.exp(-2j * np.pi * (miller @ translation))
        accumulated += coefficients[image] * phase
    return accumulated / len(rotations)
