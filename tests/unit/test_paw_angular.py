"""The angular machinery of the one-centre PAW terms: harmonics, gradients, quadrature."""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.pseudo.kb import real_spherical_harmonics
from amaterasu.pseudo.paw.angular import (
    SphericalGrid,
    angular_gradients,
    gaunt_coefficients,
    lm_index,
    real_harmonics,
)


def _random_directions(n=13, seed=0):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def test_general_harmonics_reproduce_the_hand_written_ones():
    """Two independent implementations of the same convention must agree exactly.

    The plane-wave code uses polynomials written out by hand for l <= 3; the one-centre
    code builds them from scipy for arbitrary l. If they disagree, one of the two is
    wrong, and no amount of internal consistency in either would reveal it.
    """
    directions = _random_directions()
    general = real_harmonics(3, directions)
    for l in range(4):
        hand = real_spherical_harmonics(l, directions)
        for m in range(-l, l + 1):
            assert general[lm_index(l, m)] == pytest.approx(hand[m + l], abs=1e-13)


def test_harmonics_are_orthonormal_on_the_quadrature_grid():
    grid = SphericalGrid.build(12)
    harmonics = grid.harmonics(6)
    overlap = np.einsum("k,ik,jk->ij", grid.weights, harmonics, harmonics)
    assert overlap == pytest.approx(np.eye(49), abs=1e-13)


def test_quadrature_exactness_matches_the_advertised_degree():
    """A product rule's exactness is a theorem, so it can be asserted, not just measured.

    ``exactness`` is the highest degree of a *single* harmonic the rule integrates
    exactly, so an overlap of two harmonics is only guaranteed up to half of it.
    """
    grid = SphericalGrid.of_size(4, 7)
    assert grid.exactness == 6
    lmax = grid.exactness // 2
    harmonics = grid.harmonics(lmax)
    overlap = np.einsum("k,ik,jk->ij", grid.weights, harmonics, harmonics)
    assert overlap == pytest.approx(np.eye((lmax + 1) ** 2), abs=1e-13)
    # One shell past the guarantee it is allowed to fail, and does.
    beyond = grid.harmonics(lmax + 1)
    overlap = np.einsum("k,ik,jk->ij", grid.weights, beyond, beyond)
    assert np.abs(overlap - np.eye(len(overlap))).max() > 1e-6


def test_weights_integrate_the_sphere():
    grid = SphericalGrid.build(8)
    assert float(np.sum(grid.weights)) == pytest.approx(4.0 * np.pi, rel=1e-14)


def test_angular_gradient_against_finite_differences():
    """The tangential gradient is what a GGA needs inside the sphere; check it directly."""
    directions = _random_directions(7, seed=4)
    lmax = 4
    gradients = angular_gradients(lmax, directions)
    worst = 0.0
    for k, normal in enumerate(directions):
        first = np.cross(normal, [0.3, 0.5, 0.81])
        first /= np.linalg.norm(first)
        for tangent in (first, np.cross(normal, first)):
            step = 1e-6
            plus = (normal + step * tangent) / np.linalg.norm(normal + step * tangent)
            minus = (normal - step * tangent) / np.linalg.norm(normal - step * tangent)
            reference = (
                real_harmonics(lmax, plus[None])[:, 0] - real_harmonics(lmax, minus[None])[:, 0]
            ) / (2 * step)
            worst = max(worst, float(np.abs(gradients[:, k, :] @ tangent - reference).max()))
    assert worst < 1e-8


def test_angular_gradient_is_tangential():
    directions = _random_directions(9, seed=5)
    gradients = angular_gradients(4, directions)
    radial = np.einsum("Lkc,kc->Lk", gradients, directions)
    assert np.abs(radial).max() < 1e-13


def test_gradient_refuses_the_coordinate_singularity():
    with pytest.raises(ValueError, match="z axis"):
        angular_gradients(1, np.array([[0.0, 0.0, 1.0]]))


def test_gaunt_coefficients_against_known_values():
    """Y_00 acts as the identity, and the coefficients obey the triangle rule."""
    gaunt = gaunt_coefficients(2)
    size = 9
    assert gaunt[0] == pytest.approx(np.eye(size) / np.sqrt(4.0 * np.pi), abs=1e-14)
    # <Y_L Y_i Y_j> vanishes unless l_L + l_i + l_j is even and the triangle rule holds.
    for index in range(gaunt.shape[0]):
        outer = int(np.sqrt(index))
        for i in range(size):
            li = int(np.sqrt(i))
            for j in range(size):
                lj = int(np.sqrt(j))
                forbidden = (outer + li + lj) % 2 == 1 or not (abs(li - lj) <= outer <= li + lj)
                if forbidden:
                    assert abs(gaunt[index, i, j]) < 1e-13


def test_gaunt_expansion_reproduces_a_product_of_harmonics():
    """The coefficients are exactly the expansion of Y_i Y_j in the Y_L basis."""
    gaunt = gaunt_coefficients(2)
    directions = _random_directions(11, seed=6)
    pair = real_harmonics(2, directions)
    outer = real_harmonics(4, directions)
    for i in (0, 3, 7):
        for j in (1, 4, 8):
            expansion = np.einsum("L,Lk->k", gaunt[:, i, j], outer)
            assert expansion == pytest.approx(pair[i] * pair[j], abs=1e-12)
