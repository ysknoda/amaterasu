"""Angular machinery for the one-centre (on-site) part of the PAW method.

Everything inside an augmentation sphere is a function on a sphere times a function of
the radius, so three angular tools are needed and they are all here:

**Real spherical harmonics of arbitrary degree.**
    The plane-wave side of the code needs only ``l <= 3`` and uses the hand-written
    polynomials in :mod:`amaterasu.pseudo.kb`. The on-site side needs products of two
    partial waves, which reach ``l = 2 l_max = 6``, so a general routine is required.
    Writing out six more shells by hand would be six more chances for a typo, so these
    are built from ``scipy.special.sph_harm_y`` with the standard real combination

        Y_{l,m<0} = sqrt(2) (-1)^m Im[Y_l^|m|],  Y_{l,0} = Y_l^0,
        Y_{l,m>0} = sqrt(2) (-1)^m Re[Y_l^m],

    and the test suite checks them against the hand-written ``l <= 3`` set, so the two
    independent implementations pin each other.

**Angular gradients.**
    A GGA needs ``grad n`` inside the sphere, whose tangential part is
    ``grad_Omega Y_L``. SciPy returns the theta and phi derivatives exactly (``diff_n=1``),
    which avoids both a finite difference and a hand-derived recurrence.

**Quadrature on the sphere.**
    A product grid -- Gauss-Legendre in ``cos(theta)`` times uniform in ``phi`` -- rather
    than a Lebedev grid. Lebedev is more economical for the same exactness, but its nodes
    are tabulated constants that would have to be transcribed, whereas a product rule is
    generated from ``scipy.special.roots_legendre`` and its exactness degree is a theorem:
    ``n_theta`` Gauss-Legendre nodes integrate polynomials of degree ``2 n_theta - 1`` in
    ``cos(theta)`` exactly, and ``n_phi`` equally spaced nodes integrate ``exp(i m phi)``
    exactly for ``|m| < n_phi``. :func:`SphericalGrid.exactness` reports that degree, and
    the tests confirm it by integrating harmonics up to it.

The composite index ``L = l^2 + l + m`` runs ``m = -l ... +l`` within a shell, matching
the ordering used for the Kleinman-Bylander projectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre, sph_harm_y

__all__ = [
    "SphericalGrid",
    "angular_gradients",
    "gaunt_coefficients",
    "lm_index",
    "lm_pairs",
    "real_harmonics",
]


def lm_index(l: int, m: int) -> int:
    """Composite index ``L = l^2 + l + m``."""
    return l * l + l + m


def lm_pairs(lmax: int) -> list[tuple[int, int]]:
    """``[(l, m), ...]`` in composite-index order for every shell up to ``lmax``."""
    return [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]


def _angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar and azimuthal angles of unit vectors, in SciPy's convention.

    The zero vector is mapped to ``+z`` rather than rejected, matching
    :func:`amaterasu.pseudo.kb.real_spherical_harmonics`. It occurs for the ``G = 0``
    plane wave, where the direction is multiplied by a radial factor that vanishes for
    every ``l > 0``, so the choice cannot reach a result.
    """
    unit = np.asarray(directions, dtype=float)
    norms = np.linalg.norm(unit, axis=-1, keepdims=True)
    unit = np.divide(unit, norms, out=np.zeros_like(unit), where=norms > 1e-14)
    unit[..., 2] = np.where(norms[..., 0] > 1e-14, unit[..., 2], 1.0)
    theta = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
    phi = np.arctan2(unit[..., 1], unit[..., 0])
    return theta, phi


def _combine(complex_values: np.ndarray, m: int) -> np.ndarray:
    """The real spherical harmonic built from the complex ``Y_l^|m|``."""
    if m == 0:
        return complex_values.real
    sign = (-1.0) ** m
    part = complex_values.imag if m < 0 else complex_values.real
    return np.sqrt(2.0) * sign * part


def real_harmonics(lmax: int, directions: np.ndarray) -> np.ndarray:
    """Real spherical harmonics ``Y_L`` up to ``lmax``, shape ``((lmax+1)^2, n)``.

    ``directions`` has shape ``(n, 3)`` and need not be normalised.
    """
    theta, phi = _angles(directions)
    out = np.empty(((lmax + 1) ** 2, theta.size))
    for l, m in lm_pairs(lmax):
        out[lm_index(l, m)] = _combine(sph_harm_y(l, abs(m), theta, phi), m)
    return out


def angular_gradients(lmax: int, directions: np.ndarray) -> np.ndarray:
    """Tangential gradient ``grad_Omega Y_L`` in Cartesian components.

    Returns shape ``((lmax+1)^2, n, 3)``. The result is the gradient of ``Y_L`` viewed as
    a function on the unit sphere, so it is orthogonal to the direction itself; the full
    gradient of ``Y_L(r_hat)`` at radius ``r`` is this divided by ``r``.

    The poles are a coordinate singularity of the ``(theta, phi)`` parametrisation, not of
    the field, and are excluded here: Gauss-Legendre nodes in ``cos(theta)`` never land on
    ``+-1``, so the quadrature grid never asks for one.
    """
    unit = np.asarray(directions, dtype=float)
    unit = unit / np.linalg.norm(unit, axis=-1, keepdims=True)
    theta, phi = _angles(unit)
    sin_theta = np.sin(theta)
    if np.any(np.abs(sin_theta) < 1e-12):
        raise ValueError(
            "angular_gradients was asked for a direction exactly on the z axis, where the "
            "(theta, phi) parametrisation is singular; offset the direction slightly"
        )

    cos_theta, cos_phi, sin_phi = np.cos(theta), np.cos(phi), np.sin(phi)
    # Spherical unit vectors in Cartesian components.
    theta_hat = np.stack([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], axis=-1)
    phi_hat = np.stack([-sin_phi, cos_phi, np.zeros_like(phi)], axis=-1)

    out = np.empty(((lmax + 1) ** 2, theta.size, 3))
    for l, m in lm_pairs(lmax):
        # SciPy returns (Y, D) with D[..., 0] = dY/dtheta and D[..., 1] = dY/dphi.
        _, gradient = sph_harm_y(l, abs(m), theta, phi, diff_n=1)
        real_theta = _combine(gradient[..., 0], m)
        real_phi = _combine(gradient[..., 1], m)
        out[lm_index(l, m)] = (
            real_theta[..., None] * theta_hat + (real_phi / sin_theta)[..., None] * phi_hat
        )
    return out


@dataclass(frozen=True)
class SphericalGrid:
    """A product quadrature rule on the unit sphere.

    ``weights`` sum to ``4 pi``, so ``sum_k w_k f(n_k)`` approximates ``integral f dOmega``.
    """

    directions: np.ndarray
    weights: np.ndarray
    n_theta: int
    n_phi: int

    def __len__(self) -> int:
        return len(self.weights)

    def __str__(self) -> str:
        return (
            f"spherical grid {self.n_theta}x{self.n_phi} = {len(self)} directions, "
            f"exact to degree {self.exactness}"
        )

    @property
    def exactness(self) -> int:
        """Highest degree of a single spherical harmonic this rule integrates exactly.

        An integrand built from a *product* of harmonics costs the sum of their degrees,
        so an overlap of two is guaranteed only up to ``exactness // 2`` and a Gaunt
        coefficient only up to a total of ``exactness``.
        """
        return min(2 * self.n_theta - 1, self.n_phi - 1)

    @classmethod
    def build(cls, degree: int) -> SphericalGrid:
        """The smallest product rule that is exact for harmonics up to ``degree``."""
        n_theta = (degree + 2) // 2
        n_phi = degree + 1
        return cls.of_size(max(n_theta, 1), max(n_phi, 1))

    @classmethod
    @lru_cache(maxsize=32)
    def of_size(cls, n_theta: int, n_phi: int) -> SphericalGrid:
        nodes, gauss_weights = roots_legendre(n_theta)
        cos_theta = nodes
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
        phi = 2.0 * np.pi * np.arange(n_phi) / n_phi

        directions = np.empty((n_theta * n_phi, 3))
        weights = np.empty(n_theta * n_phi)
        for i in range(n_theta):
            block = slice(i * n_phi, (i + 1) * n_phi)
            directions[block, 0] = sin_theta[i] * np.cos(phi)
            directions[block, 1] = sin_theta[i] * np.sin(phi)
            directions[block, 2] = cos_theta[i]
            weights[block] = gauss_weights[i] * (2.0 * np.pi / n_phi)
        return cls(directions=directions, weights=weights, n_theta=n_theta, n_phi=n_phi)

    def harmonics(self, lmax: int) -> np.ndarray:
        """``Y_L`` at every direction, shape ``((lmax+1)^2, npoints)``."""
        return real_harmonics(lmax, self.directions)

    def gradients(self, lmax: int) -> np.ndarray:
        """``grad_Omega Y_L`` at every direction, shape ``((lmax+1)^2, npoints, 3)``."""
        return angular_gradients(lmax, self.directions)


def gaunt_coefficients(lmax_pair: int, lmax_total: int | None = None) -> np.ndarray:
    """``G[L, i, j] = integral Y_L Y_i Y_j dOmega`` for real spherical harmonics.

    ``i`` and ``j`` run over shells up to ``lmax_pair`` and ``L`` up to ``lmax_total``
    (default ``2 lmax_pair``, which is where the product of two harmonics stops).

    Evaluated by quadrature rather than from a closed form: the integrand is a polynomial
    of degree ``2 lmax_pair + lmax_total`` on the sphere, so a rule exact to that degree
    gives the coefficients to machine precision -- a computation, not an approximation.
    """
    total = 2 * lmax_pair if lmax_total is None else lmax_total
    grid = SphericalGrid.build(2 * lmax_pair + total)
    pair = grid.harmonics(lmax_pair)
    outer = grid.harmonics(total)
    return np.einsum("k,Lk,ik,jk->Lij", grid.weights, outer, pair, pair)


@lru_cache(maxsize=16)
def solid_harmonic_polynomial(l: int) -> tuple[np.ndarray, np.ndarray]:
    """Monomial exponents and coefficients of ``C_lm(q) = |q|^l Y_lm(q_hat)``.

    Returns ``(exponents, coefficients)`` with shapes ``(nmonomials, 3)`` and
    ``(2l+1, nmonomials)``, so that
    ``C_lm(q) = sum_n coefficients[m, n] q_x^e0 q_y^e1 q_z^e2``.

    The solid harmonics are homogeneous polynomials of degree ``l``, so the coefficients
    are determined *exactly* by evaluating them at as many points as there are monomials
    and solving the resulting linear system -- an identification, not a fit.

    Why not just differentiate ``Y_lm`` in spherical coordinates: because ``1/sin(theta)``
    is singular on the ``z`` axis, and the ``z`` axis is exactly where a reciprocal-lattice
    vector often sits. The polynomial form has no such coordinate and its gradient is
    elementary; :mod:`amaterasu.pseudo.kb` writes the same objects out by hand for
    ``l <= 3`` and the test suite checks the two against each other.
    """
    exponents = np.array(
        [(i, j, l - i - j) for i in range(l + 1) for j in range(l - i + 1)], dtype=int
    )
    rng = np.random.default_rng(20240501 + l)
    points = rng.normal(size=(len(exponents) * 3, 3))
    monomials = np.prod(points[:, None, :] ** exponents[None, :, :], axis=-1)
    radii = np.linalg.norm(points, axis=1)
    values = real_harmonics(l, points)[l**2 :] * radii**l
    coefficients, *_ = np.linalg.lstsq(monomials, values.T, rcond=None)
    residual = np.abs(monomials @ coefficients - values.T).max()
    if residual > 1e-9:  # pragma: no cover - would mean the harmonics are not polynomial
        raise RuntimeError(f"solid harmonics of degree {l} did not fit a polynomial")
    return exponents, np.ascontiguousarray(coefficients.T)


def solid_harmonics_any_l(l: int, vectors: np.ndarray) -> np.ndarray:
    """``C_lm(q)`` for arbitrary ``l``, shape ``(2l+1, n)``."""
    exponents, coefficients = solid_harmonic_polynomial(l)
    q = np.atleast_2d(np.asarray(vectors, dtype=float))
    monomials = np.prod(q[:, None, :] ** exponents[None, :, :], axis=-1)
    return coefficients @ monomials.T


def solid_harmonics_gradient_any_l(l: int, vectors: np.ndarray) -> np.ndarray:
    """``dC_lm/dq_alpha`` for arbitrary ``l``, shape ``(2l+1, n, 3)``."""
    exponents, coefficients = solid_harmonic_polynomial(l)
    q = np.atleast_2d(np.asarray(vectors, dtype=float))
    out = np.zeros((len(coefficients), len(q), 3))
    for axis in range(3):
        power = exponents[exponents[:, axis] > 0]
        if len(power) == 0:
            continue
        keep = exponents[:, axis] > 0
        reduced = exponents[keep].copy()
        factor = reduced[:, axis].astype(float)
        reduced[:, axis] -= 1
        monomials = np.prod(q[:, None, :] ** reduced[None, :, :], axis=-1)
        out[..., axis] = (coefficients[:, keep] * factor) @ monomials.T
    return out
