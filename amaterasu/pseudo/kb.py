"""Kleinman-Bylander projectors in the plane-wave basis.

The separable nonlocal pseudopotential is

    V_NL = sum_a sum_{ij} |beta^a_i> D^a_{ij} <beta^a_j|,

with ``beta^a_{lmn}(r) = beta_{ln}(|r - tau_a|) Y_lm(r - tau_a)`` built from real
spherical harmonics. Because the plane wave ``|k+G> = exp(i(k+G).r)/sqrt(Omega)`` and the
projector are both known analytically once the radial transform ``f_l(q)`` is in hand, the
matrix element is closed-form:

    <k+G|beta^a_{lmn}> = (4 pi / sqrt(Omega)) (-i)^l Y_lm(k+G) f_ln(|k+G|) exp(-i(k+G).tau_a)

This module builds that matrix for one k-point and applies the operator to a block of
bands. Two properties make it testable without a reference code:

* **completeness** -- ``sum_G |<k+G|beta>|^2 -> 1`` as the cutoff grows, because the
  generators normalise ``integral (r beta)^2 dr = 1`` and the Hankel transform preserves
  that norm. How close it gets at a working cutoff is a direct measure of how well the
  basis resolves the projector.
* **rotational invariance** -- ``sum_m |Y_lm|^2 = (2l+1)/(4 pi)`` is a constant, so a
  filled ``l`` shell contributes isotropically. The addition theorem behind it is checked
  in the test suite, which pins the harmonics without relying on a phase convention.

Only the ``m`` ordering is a convention (``m = -l ... +l`` here). It cannot affect a
pseudopotential energy, where each ``l`` shell enters as a whole, but it will matter for
the DFT+U occupation matrix, so it is fixed and documented rather than left implicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import NCPseudopotential
from .formfactors import projector_form_factor

__all__ = [
    "KBProjectors",
    "build_projectors",
    "real_spherical_harmonics",
    "solid_harmonics",
    "solid_harmonics_gradient",
]

#: Highest angular momentum implemented. PseudoDojo tables reach l = 3 (f) for the
#: lanthanides and actinides; nothing in the tables goes beyond that.
MAX_L = 3


def real_spherical_harmonics(l: int, directions: np.ndarray) -> np.ndarray:
    """Real spherical harmonics ``Y_lm`` for ``m = -l ... +l`` on unit vectors.

    ``directions`` has shape ``(n, 3)`` and need not be normalised; the zero vector is
    mapped to ``+z`` so that ``q = 0`` stays finite (it is multiplied by ``f_l(0) = 0``
    for every ``l > 0``, so the choice cannot influence a result).

    Returns an array of shape ``(2l+1, n)``.
    """
    if not 0 <= l <= MAX_L:
        raise NotImplementedError(f"real spherical harmonics are implemented up to l={MAX_L}")
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    norms = np.linalg.norm(directions, axis=1)
    unit = np.zeros_like(directions)
    unit[:, 2] = 1.0
    nonzero = norms > 1e-14
    unit[nonzero] = directions[nonzero] / norms[nonzero, None]
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]

    if l == 0:
        return np.full((1, len(unit)), 0.5 / np.sqrt(np.pi))
    if l == 1:
        c = np.sqrt(0.75 / np.pi)
        return c * np.stack([y, z, x])
    if l == 2:
        c2 = 0.5 * np.sqrt(15.0 / np.pi)
        return np.stack(
            [
                c2 * x * y,
                c2 * y * z,
                0.25 * np.sqrt(5.0 / np.pi) * (3.0 * z**2 - 1.0),
                c2 * x * z,
                0.25 * np.sqrt(15.0 / np.pi) * (x**2 - y**2),
            ]
        )
    return np.stack(
        [
            0.25 * np.sqrt(17.5 / np.pi) * y * (3.0 * x**2 - y**2),
            0.5 * np.sqrt(105.0 / np.pi) * x * y * z,
            0.25 * np.sqrt(10.5 / np.pi) * y * (5.0 * z**2 - 1.0),
            0.25 * np.sqrt(7.0 / np.pi) * z * (5.0 * z**2 - 3.0),
            0.25 * np.sqrt(10.5 / np.pi) * x * (5.0 * z**2 - 1.0),
            0.25 * np.sqrt(105.0 / np.pi) * z * (x**2 - y**2),
            0.25 * np.sqrt(17.5 / np.pi) * x * (x**2 - 3.0 * y**2),
        ]
    )


def solid_harmonics(l: int, vectors: np.ndarray) -> np.ndarray:
    """Real *solid* harmonics ``C_lm(q) = |q|^l Y_lm(q_hat)``, shape ``(2l+1, n)``.

    These are homogeneous polynomials of degree ``l`` in the Cartesian components, which
    is what makes their gradient elementary -- and the gradient is what the nonlocal
    stress needs, since straining the cell rotates and stretches every ``k+G``.
    """
    if not 0 <= l <= MAX_L:
        raise NotImplementedError(f"solid harmonics are implemented up to l={MAX_L}")
    q = np.atleast_2d(np.asarray(vectors, dtype=float))
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    r2 = x**2 + y**2 + z**2

    if l == 0:
        return np.full((1, len(q)), 0.5 / np.sqrt(np.pi))
    if l == 1:
        c = np.sqrt(0.75 / np.pi)
        return c * np.stack([y, z, x])
    if l == 2:
        c = 0.5 * np.sqrt(15.0 / np.pi)
        return np.stack(
            [
                c * x * y,
                c * y * z,
                0.25 * np.sqrt(5.0 / np.pi) * (3.0 * z**2 - r2),
                c * x * z,
                0.25 * np.sqrt(15.0 / np.pi) * (x**2 - y**2),
            ]
        )
    return np.stack(
        [
            0.25 * np.sqrt(17.5 / np.pi) * y * (3.0 * x**2 - y**2),
            0.5 * np.sqrt(105.0 / np.pi) * x * y * z,
            0.25 * np.sqrt(10.5 / np.pi) * y * (5.0 * z**2 - r2),
            0.25 * np.sqrt(7.0 / np.pi) * z * (5.0 * z**2 - 3.0 * r2),
            0.25 * np.sqrt(10.5 / np.pi) * x * (5.0 * z**2 - r2),
            0.25 * np.sqrt(105.0 / np.pi) * z * (x**2 - y**2),
            0.25 * np.sqrt(17.5 / np.pi) * x * (x**2 - 3.0 * y**2),
        ]
    )


def solid_harmonics_gradient(l: int, vectors: np.ndarray) -> np.ndarray:
    """``dC_lm/dq_alpha``, shape ``(2l+1, n, 3)``.

    Written out rather than differentiated numerically, and checked against a finite
    difference of :func:`solid_harmonics` in the test suite -- a polynomial's derivative
    is exact, so any discrepancy there is a typo rather than a tolerance.
    """
    if not 0 <= l <= MAX_L:
        raise NotImplementedError(f"solid harmonics are implemented up to l={MAX_L}")
    q = np.atleast_2d(np.asarray(vectors, dtype=float))
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    r2 = x**2 + y**2 + z**2
    zero = np.zeros_like(x)

    if l == 0:
        return np.zeros((1, len(q), 3))
    if l == 1:
        c = np.sqrt(0.75 / np.pi)
        one, nil = np.ones_like(x), zero
        return c * np.stack(
            [
                np.stack([nil, one, nil], axis=-1),
                np.stack([nil, nil, one], axis=-1),
                np.stack([one, nil, nil], axis=-1),
            ]
        )
    if l == 2:
        c = 0.5 * np.sqrt(15.0 / np.pi)
        cz = 0.25 * np.sqrt(5.0 / np.pi)
        cd = 0.25 * np.sqrt(15.0 / np.pi)
        return np.stack(
            [
                c * np.stack([y, x, zero], axis=-1),
                c * np.stack([zero, z, y], axis=-1),
                cz * np.stack([-2.0 * x, -2.0 * y, 4.0 * z], axis=-1),
                c * np.stack([z, zero, x], axis=-1),
                cd * np.stack([2.0 * x, -2.0 * y, zero], axis=-1),
            ]
        )
    ca = 0.25 * np.sqrt(17.5 / np.pi)
    cb = 0.5 * np.sqrt(105.0 / np.pi)
    cc = 0.25 * np.sqrt(10.5 / np.pi)
    cd = 0.25 * np.sqrt(7.0 / np.pi)
    ce = 0.25 * np.sqrt(105.0 / np.pi)
    return np.stack(
        [
            ca * np.stack([6.0 * x * y, 3.0 * (x**2 - y**2), zero], axis=-1),
            cb * np.stack([y * z, x * z, x * y], axis=-1),
            cc * np.stack([-2.0 * x * y, 5.0 * z**2 - r2 - 2.0 * y**2, 8.0 * y * z], axis=-1),
            cd * np.stack([-6.0 * x * z, -6.0 * y * z, 9.0 * z**2 - 3.0 * r2], axis=-1),
            cc * np.stack([5.0 * z**2 - r2 - 2.0 * x**2, -2.0 * x * y, 8.0 * x * z], axis=-1),
            ce * np.stack([2.0 * x * z, -2.0 * y * z, x**2 - y**2], axis=-1),
            ca * np.stack([3.0 * (x**2 - y**2), -6.0 * x * y, zero], axis=-1),
        ]
    )


@dataclass
class KBProjectors:
    """The nonlocal operator at one k-point, as a dense projector matrix.

    ``matrix`` is ``<k+G|beta_i>`` with shape ``(nproj, npw)`` and ``dij`` the coupling
    ``D_ij`` in Hartree with shape ``(nproj, nproj)``. Projectors are laid out atom by
    atom, and within an atom by radial channel and then ``m``.
    """

    matrix: np.ndarray
    dij: np.ndarray
    #: ``(atom index, l, radial channel index, m)`` for each row, for diagnostics and DFT+U.
    labels: list[tuple[int, int, int, int]]

    def __len__(self) -> int:
        return len(self.labels)

    def project(self, coefficients: np.ndarray) -> np.ndarray:
        """``<beta_i|psi_n>`` for a band or a block of bands, shape ``(..., nproj)``."""
        return np.asarray(coefficients) @ self.matrix.conj().T

    def apply(self, coefficients: np.ndarray, coupling: np.ndarray | None = None) -> np.ndarray:
        """``V_NL |psi>`` in the plane-wave basis, same shape as ``coefficients``.

        ``coupling`` overrides the stored ``D_ij``. A norm-conserving dataset's coupling is
        a constant of the file; a PAW one is ``dE/dD_ij``, rebuilt from the current
        potential every SCF iteration, and is passed in here.
        """
        matrix = self.dij if coupling is None else coupling
        if len(self.labels) == 0:
            return np.zeros_like(coefficients)
        return (self.project(coefficients) @ matrix) @ self.matrix

    def energy(
        self, coefficients: np.ndarray, weights: np.ndarray, coupling: np.ndarray | None = None
    ) -> float:
        """``sum_n w_n <psi_n|V_NL|psi_n>`` in Hartree."""
        matrix = self.dij if coupling is None else coupling
        if len(self.labels) == 0:
            return 0.0
        projections = self.project(coefficients)
        quadratic = np.einsum("ni,ij,nj->n", projections.conj(), matrix, projections).real
        return float(np.dot(weights, quadratic))

    def occupancies(self, coefficients: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """``D_ij = sum_n w_n <p_i|psi_n><psi_n|p_j>``, the PAW atomic density matrix.

        Real by time-reversal symmetry: ``k`` and ``-k`` contribute complex conjugates, so
        the imaginary part cancels over the full zone. Taking the real part here is what
        supplies that cancellation when the sum runs over the irreducible wedge instead.
        """
        if len(self.labels) == 0:
            return np.zeros((0, 0))
        projections = self.project(coefficients)
        return np.einsum("n,ni,nj->ij", weights, projections, projections.conj()).real

    def completeness(self) -> np.ndarray:
        """``sum_G |<k+G|beta_i>|^2`` per projector; approaches 1 as the cutoff grows."""
        return np.einsum("ig,ig->i", self.matrix.conj(), self.matrix).real


def build_projectors(
    sphere,
    positions: np.ndarray,
    symbols: list[str],
    pseudopotentials: dict[str, NCPseudopotential],
    volume: float,
) -> KBProjectors:
    """Assemble the KB projector matrix for the k-point ``sphere`` was built for.

    Parameters
    ----------
    sphere
        The wavefunction :class:`~amaterasu.basis.gvectors.GSphere` for this k-point.
    positions
        Cartesian atomic positions in Bohr, shape ``(natoms, 3)``.
    symbols
        Chemical symbol of each atom, in the same order as ``positions``.
    pseudopotentials
        One dataset per distinct symbol.
    volume
        Cell volume in Bohr^3.
    """
    kg = sphere.cartesian
    q = np.sqrt(sphere.g2)
    prefactor = 4.0 * np.pi / np.sqrt(volume)

    # f_l(|k+G|) depends only on the radial data, so it is computed once per species and
    # reused for every atom of that species and every m.
    form_factors: dict[tuple[str, int, int], np.ndarray] = {}
    harmonics: dict[int, np.ndarray] = {}
    for symbol, pseudo in pseudopotentials.items():
        for projector in pseudo.projectors:
            key = (symbol, projector.l, projector.index)
            form_factors[key] = projector_form_factor(pseudo.grid, projector, q)
            if projector.l not in harmonics:
                harmonics[projector.l] = real_spherical_harmonics(projector.l, kg)

    rows: list[np.ndarray] = []
    labels: list[tuple[int, int, int, int]] = []
    blocks: list[np.ndarray] = []
    for atom, symbol in enumerate(symbols):
        pseudo = pseudopotentials[symbol]
        phase = np.exp(-1j * (kg @ positions[atom]))
        channels = pseudo.projectors
        # D_ij for this atom, expanded over m. Same-l channels couple; different l do not.
        atom_rows: list[np.ndarray] = []
        atom_labels: list[tuple[int, int, int, int]] = []
        for index, projector in enumerate(channels):
            radial = form_factors[(symbol, projector.l, projector.index)]
            angular = harmonics[projector.l]
            for m_index in range(2 * projector.l + 1):
                atom_rows.append(
                    prefactor * (-1j) ** projector.l * angular[m_index] * radial * phase
                )
                atom_labels.append((atom, projector.l, index, m_index - projector.l))
        size = len(atom_rows)
        block = np.zeros((size, size))
        for row, (_, l_row, ch_row, m_row) in enumerate(atom_labels):
            for column, (_, l_col, ch_col, m_col) in enumerate(atom_labels):
                if l_row != l_col or m_row != m_col:
                    continue
                if pseudo.dij is not None:
                    block[row, column] = pseudo.dij[ch_row, ch_col]
                elif ch_row == ch_col:
                    block[row, column] = channels[ch_row].energy
        rows.extend(atom_rows)
        labels.extend(atom_labels)
        blocks.append(block)

    if not rows:
        return KBProjectors(
            matrix=np.zeros((0, len(sphere)), dtype=complex),
            dij=np.zeros((0, 0)),
            labels=[],
        )

    total = sum(len(b) for b in blocks)
    dij = np.zeros((total, total))
    offset = 0
    for block in blocks:
        size = len(block)
        dij[offset : offset + size, offset : offset + size] = block
        offset += size
    return KBProjectors(matrix=np.asarray(rows), dij=dij, labels=labels)
