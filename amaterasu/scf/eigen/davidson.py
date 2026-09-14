"""Block Davidson diagonalisation of the Kohn-Sham Hamiltonian at one k-point.

The Hamiltonian is never formed: only its action on a block of bands is needed, which is
one batched FFT pair plus a projector contraction. Davidson's method builds a small
subspace spanned by the current Ritz vectors and their *preconditioned* residuals,
diagonalises ``H`` exactly inside it, and repeats. The preconditioner
(:mod:`amaterasu.scf.eigen.precond`) is what turns this from a slow power-like iteration
into a fast one, because it flattens the kinetic-energy spread that otherwise dominates
every residual.

The subspace is collapsed back to the Ritz vectors once it grows past
``subspace_factor * nbands``: the small eigenproblem is dense and cubic in the subspace
size, and a stale search direction contributes nothing anyway.

Only converged residual norms are reported, not a claim about eigenvalue accuracy. The
two are related by ``|delta eps| <= ||r||`` for a Hermitian operator, and by the much
tighter ``||r||^2 / gap`` when the level is well separated, so a residual threshold is a
safe (conservative) proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .precond import teter_payne_allan
from .subspace import orthogonalize_against, orthonormalize, rayleigh_ritz

__all__ = ["DavidsonResult", "davidson"]


@dataclass
class DavidsonResult:
    """Eigenpairs at one k-point, with the diagnostics needed to trust them."""

    eigenvalues: np.ndarray
    coefficients: np.ndarray
    residual_norms: np.ndarray
    iterations: int
    converged: bool

    @property
    def nbands(self) -> int:
        return len(self.eigenvalues)


def davidson(
    apply_hamiltonian,
    guess: np.ndarray,
    kinetic: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 40,
    subspace_factor: int = 4,
) -> DavidsonResult:
    """Find the lowest eigenpairs of a Hermitian operator given a starting block.

    Parameters
    ----------
    apply_hamiltonian
        Callable mapping a block of shape ``(m, npw)`` to ``H`` times that block.
    guess
        Starting bands, shape ``(nbands, npw)``; need not be orthonormal.
    kinetic
        ``|k+G|^2 / 2`` for this k-point, used by the preconditioner.
    tolerance
        Convergence threshold on the residual norm of every band, in Hartree.
    """
    basis = orthonormalize(np.asarray(guess, dtype=complex))
    nbands, npw = basis.shape
    max_subspace = min(max(subspace_factor, 2) * nbands, npw)

    h_basis = apply_hamiltonian(basis)
    eigenvalues, vectors, h_vectors = rayleigh_ritz(basis, h_basis, nbands)
    norms = np.full(nbands, np.inf)

    iteration = 0
    for _pass in range(1, max_iterations + 1):
        iteration = _pass
        residual = h_vectors - eigenvalues[:, None] * vectors
        norms = np.linalg.norm(residual, axis=1)
        active = norms > tolerance
        if not np.any(active):
            break

        band_kinetic = np.einsum("ng,g->n", np.abs(vectors[active]) ** 2, kinetic)
        correction = teter_payne_allan(residual[active], kinetic, band_kinetic)
        # Normalise before orthogonalising: the drop threshold in orthogonalize_against
        # measures linear dependence, and an un-normalised correction whose norm has
        # fallen to the residual tolerance would be discarded as "dependent" purely
        # because it is small, stalling the iteration exactly where it should refine.
        correction /= np.maximum(np.linalg.norm(correction, axis=1, keepdims=True), 1e-300)
        new = orthogonalize_against(correction, basis)
        if len(new) == 0:  # pragma: no cover - the subspace already spans the corrections
            break

        basis = np.vstack([basis, new])
        h_basis = np.vstack([h_basis, apply_hamiltonian(new)])
        eigenvalues, vectors, h_vectors = rayleigh_ritz(basis, h_basis, nbands)

        if len(basis) + nbands > max_subspace:
            basis, h_basis = vectors, h_vectors

    return DavidsonResult(
        eigenvalues=eigenvalues,
        coefficients=vectors,
        residual_norms=norms,
        iterations=iteration,
        converged=bool(np.all(norms <= tolerance)),
    )


def generalized_davidson(
    apply_hamiltonian,
    apply_overlap,
    guess: np.ndarray,
    kinetic: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 40,
    subspace_factor: int = 4,
) -> DavidsonResult:
    """Block Davidson for ``H c = eps S c``, which is what PAW's overlap operator makes it.

    Identical in structure to :func:`davidson`, with three differences: the subspace is
    kept orthonormal in the ``S`` metric rather than the Euclidean one, the residual is
    ``H c - eps S c``, and ``S`` applied to the basis is carried along so it never has to
    be recomputed. ``S`` is a low-rank update of the identity, so applying it costs one
    projector contraction -- far less than applying ``H``.
    """
    from .subspace import generalized_rayleigh_ritz, s_orthonormalize, s_project_out

    block = np.asarray(guess, dtype=complex)
    basis, s_basis = s_orthonormalize(block, apply_overlap(block))
    nbands, npw = basis.shape
    max_subspace = min(max(subspace_factor, 2) * nbands, npw)

    h_basis = apply_hamiltonian(basis)
    eigenvalues, vectors, h_vectors, s_vectors = generalized_rayleigh_ritz(
        basis, h_basis, s_basis, nbands
    )
    norms = np.full(nbands, np.inf)

    iteration = 0
    for _pass in range(1, max_iterations + 1):
        iteration = _pass
        residual = h_vectors - eigenvalues[:, None] * s_vectors
        norms = np.linalg.norm(residual, axis=1)
        active = norms > tolerance
        if not np.any(active):
            break

        band_kinetic = np.einsum("ng,g->n", np.abs(vectors[active]) ** 2, kinetic)
        correction = teter_payne_allan(residual[active], kinetic, band_kinetic)
        correction /= np.maximum(np.linalg.norm(correction, axis=1, keepdims=True), 1e-300)
        new = s_project_out(correction, basis, s_basis)
        keep = np.linalg.norm(new, axis=1) > 1e-7
        new = new[keep]
        if len(new) == 0:  # pragma: no cover - the subspace already spans the corrections
            break
        new, s_new = s_orthonormalize(new, apply_overlap(new))

        basis = np.vstack([basis, new])
        s_basis = np.vstack([s_basis, s_new])
        h_basis = np.vstack([h_basis, apply_hamiltonian(new)])
        eigenvalues, vectors, h_vectors, s_vectors = generalized_rayleigh_ritz(
            basis, h_basis, s_basis, nbands
        )

        if len(basis) + nbands > max_subspace:
            basis, h_basis, s_basis = vectors, h_vectors, s_vectors

    return DavidsonResult(
        eigenvalues=eigenvalues,
        coefficients=vectors,
        residual_norms=norms,
        iterations=iteration,
        converged=bool(np.all(norms <= tolerance)),
    )
