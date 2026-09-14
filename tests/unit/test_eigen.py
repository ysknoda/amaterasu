"""Iterative diagonalisation: Davidson, the preconditioner and the subspace utilities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

from amaterasu.scf.eigen import (
    davidson,
    orthogonalize_against,
    orthonormalize,
    random_bands,
    rayleigh_ritz,
    solve,
    teter_payne_allan,
)


def _hermitian_operator(size=240, spread=40.0, coupling=0.02, seed=0):
    """A kinetic-like diagonal plus a random Hermitian perturbation."""
    rng = np.random.default_rng(seed)
    diagonal = np.sort(rng.uniform(0.0, spread, size))
    off = coupling * (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
    return np.diag(diagonal) + off + off.conj().T, diagonal


def test_davidson_matches_dense_diagonalisation():
    """The lowest eigenvalues must agree with a direct dense solve to near machine precision."""
    matrix, diagonal = _hermitian_operator()
    reference = eigh(matrix, eigvals_only=True)[:8]
    result = davidson(
        lambda block: block @ matrix.T,
        random_bands(len(diagonal), 8, diagonal, seed=1),
        diagonal,
        tolerance=1e-10,
    )
    assert result.converged
    assert result.eigenvalues == pytest.approx(reference, abs=1e-11)
    overlap = result.coefficients @ result.coefficients.conj().T
    assert overlap == pytest.approx(np.eye(8), abs=1e-12)


def test_davidson_residuals_bound_the_eigenvalue_error():
    """|delta eps| <= ||r|| for a Hermitian operator, so the reported norms are meaningful."""
    matrix, diagonal = _hermitian_operator(seed=4)
    reference = eigh(matrix, eigvals_only=True)[:5]
    result = davidson(
        lambda block: block @ matrix.T,
        random_bands(len(diagonal), 5, diagonal, seed=2),
        diagonal,
        tolerance=1e-5,
        max_iterations=6,
    )
    assert np.all(np.abs(result.eigenvalues - reference) <= result.residual_norms + 1e-14)


def test_davidson_converges_faster_from_a_good_guess():
    matrix, diagonal = _hermitian_operator(seed=7)
    cold = davidson(
        lambda b: b @ matrix.T,
        random_bands(len(diagonal), 6, diagonal, seed=3),
        diagonal,
        tolerance=1e-10,
    )
    warm = davidson(lambda b: b @ matrix.T, cold.coefficients, diagonal, tolerance=1e-10)
    assert warm.iterations < cold.iterations
    assert warm.eigenvalues == pytest.approx(cold.eigenvalues, abs=1e-12)


def test_preconditioner_limits():
    """K -> 1 at small kinetic energy and falls off as 1/(2x) at large."""
    kinetic = np.array([0.0, 1.0, 1000.0])
    residual = np.ones((1, 3))
    factor = teter_payne_allan(residual, kinetic, np.array([1.0]))[0]
    assert factor[0] == pytest.approx(1.0)
    assert factor[1] < 1.0
    assert factor[2] == pytest.approx(1.0 / (2.0 * 1000.0), rel=2e-3)
    assert np.all(np.diff(factor) < 0)


def test_preconditioner_survives_a_zero_reference():
    factor = teter_payne_allan(np.ones((1, 2)), np.array([1.0, 2.0]), np.array([0.0]))
    assert np.all(np.isfinite(factor))


def test_orthonormalize_and_project():
    rng = np.random.default_rng(0)
    block = rng.normal(size=(5, 30)) + 1j * rng.normal(size=(5, 30))
    orthonormal = orthonormalize(block)
    assert orthonormal @ orthonormal.conj().T == pytest.approx(np.eye(5), abs=1e-12)

    extra = rng.normal(size=(3, 30)) + 1j * rng.normal(size=(3, 30))
    new = orthogonalize_against(extra, orthonormal)
    assert new @ orthonormal.conj().T == pytest.approx(np.zeros((len(new), 5)), abs=1e-12)
    assert new @ new.conj().T == pytest.approx(np.eye(len(new)), abs=1e-12)


def test_orthogonalize_drops_dependent_directions():
    """A direction already inside the basis carries no information and must be discarded."""
    rng = np.random.default_rng(1)
    basis = orthonormalize(rng.normal(size=(4, 20)))
    duplicate = basis[:2] + 1e-14 * rng.normal(size=(2, 20))
    assert len(orthogonalize_against(duplicate, basis)) == 0


def test_rayleigh_ritz_is_exact_in_a_complete_subspace():
    matrix, _ = _hermitian_operator(size=30, seed=9)
    basis = orthonormalize(np.eye(30, dtype=complex))
    values, vectors, h_vectors = rayleigh_ritz(basis, basis @ matrix.T, 30)
    assert values == pytest.approx(eigh(matrix, eigvals_only=True), abs=1e-12)
    assert h_vectors == pytest.approx(vectors @ matrix.T, abs=1e-11)


def test_random_bands_are_orthonormal_and_damped():
    kinetic = np.linspace(0.0, 50.0, 200)
    block = random_bands(200, 6, kinetic, seed=0)
    assert block @ block.conj().T == pytest.approx(np.eye(6), abs=1e-12)
    # Weight must sit at low G, not spread evenly over the sphere.
    weight = np.mean(np.abs(block) ** 2, axis=0)
    assert weight[:20].sum() > 5.0 * weight[-20:].sum()


def test_random_bands_are_reproducible():
    kinetic = np.linspace(0.0, 10.0, 50)
    assert random_bands(50, 3, kinetic, seed=5) == pytest.approx(
        random_bands(50, 3, kinetic, seed=5)
    )


def test_solve_rejects_an_unimplemented_eigensolver():
    kinetic = np.linspace(0.1, 5.0, 20)
    with pytest.raises(NotImplementedError, match="lobpcg"):
        solve("lobpcg", lambda b: b, random_bands(20, 2, kinetic), kinetic)
