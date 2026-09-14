"""Subspace utilities shared by the iterative eigensolvers.

Bands are stored as rows: a block of shape ``(nbands, npw)``. Orthonormality means
``sum_G psi_i(G)* psi_j(G) = delta_ij``, and every routine here preserves it, because the
Rayleigh-Ritz step below assumes it (it solves a standard rather than a generalised
eigenproblem, which is both faster and better conditioned).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular

__all__ = ["orthogonalize_against", "orthonormalize", "random_bands", "rayleigh_ritz"]


def orthonormalize(block: np.ndarray) -> np.ndarray:
    """Orthonormalise the rows of ``block`` (Loewdin-free: a Cholesky of the overlap).

    Falls back to an SVD when the overlap is too ill-conditioned for a Cholesky, which
    happens if two starting vectors are nearly parallel.
    """
    block = np.asarray(block, dtype=complex)
    overlap = block @ block.conj().T
    try:
        factor = cholesky(overlap, lower=True)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return _svd_orthonormalize(block)
    result = solve_triangular(factor, block, lower=True)
    if not np.all(np.isfinite(result)):  # pragma: no cover - defensive
        return _svd_orthonormalize(block)
    return result


def _svd_orthonormalize(block: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    _, singular, right = np.linalg.svd(block, full_matrices=False)
    keep = singular > threshold * max(float(singular[0]), 1e-300)
    return right[keep]


def orthogonalize_against(
    block: np.ndarray, basis: np.ndarray, *, threshold: float = 1e-7
) -> np.ndarray:
    """Project ``block`` off ``basis`` and orthonormalise what is left.

    The projection is applied twice: one pass leaves a residual of order the machine
    epsilon divided by the smallest singular value, which for a nearly dependent search
    direction is not small. Directions whose remaining norm falls below ``threshold`` are
    dropped rather than normalised, since amplifying numerical noise into the subspace is
    how a Davidson iteration stalls.
    """
    block = np.asarray(block, dtype=complex)
    for _ in range(2):
        block = block - (block @ basis.conj().T) @ basis
    norms = np.linalg.norm(block, axis=1)
    block = block[norms > threshold]
    if len(block) == 0:
        return block
    _, singular, right = np.linalg.svd(block, full_matrices=False)
    return right[singular > threshold * float(singular[0])]


def rayleigh_ritz(
    basis: np.ndarray, h_basis: np.ndarray, nbands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalise ``H`` inside an orthonormal subspace.

    Returns the ``nbands`` lowest eigenvalues, the corresponding Ritz vectors, and
    ``H`` applied to them -- the last for free, since it is the same linear combination
    of the already-computed ``h_basis``.
    """
    projected = basis.conj() @ h_basis.T
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors = eigh(projected)
    rotation = vectors[:, :nbands].T
    return values[:nbands], rotation @ basis, rotation @ h_basis


def random_bands(npw: int, nbands: int, kinetic: np.ndarray, seed: int = 0) -> np.ndarray:
    """A reproducible random starting block, damped by ``1/(1 + T_G)``.

    A white-noise start puts as much weight on the highest plane wave as on the lowest,
    which no physical band has; the damping gives the first Rayleigh-Ritz step something
    with roughly the right envelope and typically saves several iterations.
    """
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(nbands, npw)) + 1j * rng.normal(size=(nbands, npw))
    return orthonormalize(block / (1.0 + kinetic))


# --------------------------------------------------------------------- generalised
# PAW makes the smooth orbitals non-orthogonal: <psi~|S|psi~> = delta with
# S = 1 + sum_a |p_i> O_ij <p_j>. These routines are the S-weighted versions of the ones
# above. They are kept separate rather than folded in with a default of S = 1, because the
# norm-conserving path is well tested and there is no reason to disturb it.


def s_orthonormalize(block: np.ndarray, s_block: np.ndarray):
    """Orthonormalise rows so that ``conj(block) . s_block^T = I``; returns both blocks.

    ``s_block`` must be ``S`` applied to ``block``. Both are transformed by the same
    triangular factor, so the caller never has to apply ``S`` twice.
    """
    overlap = block @ s_block.conj().T
    overlap = 0.5 * (overlap + overlap.conj().T)
    try:
        factor = cholesky(overlap, lower=True)
    except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - defensive
        values, vectors = eigh(overlap)
        keep = values > 1e-12 * max(float(values[-1]), 1e-300)
        rotation = (vectors[:, keep] / np.sqrt(values[keep])).T
        return rotation @ block, rotation @ s_block
    return (
        solve_triangular(factor, block, lower=True),
        solve_triangular(factor, s_block, lower=True),
    )


def s_project_out(block: np.ndarray, basis: np.ndarray, s_basis: np.ndarray) -> np.ndarray:
    """Remove the part of ``block`` that lies in ``basis``, in the ``S`` metric.

    Applied twice for the same reason as the Euclidean version: one pass leaves a residual
    of order the machine epsilon over the smallest singular value.
    """
    for _ in range(2):
        block = block - (block @ s_basis.conj().T) @ basis
    return block


def generalized_rayleigh_ritz(
    basis: np.ndarray, h_basis: np.ndarray, s_basis: np.ndarray, nbands: int
):
    """Diagonalise ``H`` in an ``S``-orthonormal subspace.

    Because the basis is already ``S``-orthonormal the small problem is a standard one,
    which is both faster and better conditioned than a generalised solve on a subspace
    that may be nearly dependent.
    """
    projected = basis.conj() @ h_basis.T
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors = eigh(projected)
    rotation = vectors[:, :nbands].T
    return values[:nbands], rotation @ basis, rotation @ h_basis, rotation @ s_basis
