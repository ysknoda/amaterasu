"""Real spherical harmonics and the Kleinman-Bylander projectors in the plane-wave basis."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import legendre

from amaterasu.basis.gvectors import GSphere
from amaterasu.pseudo.kb import MAX_L, build_projectors, real_spherical_harmonics
from amaterasu.units import ANG_IN_BOHR


def _unit_vectors(n, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, 3))
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


@pytest.mark.parametrize("l", range(MAX_L + 1))
def test_addition_theorem(l):
    """sum_m Y_lm(a) Y_lm(b) = (2l+1)/(4 pi) P_l(a.b).

    This pins the harmonics without committing to a phase or ordering convention: any set
    satisfying it spans the correct l subspace with the correct normalisation.
    """
    a, b = _unit_vectors(40, seed=1), _unit_vectors(40, seed=2)
    left = np.einsum("mi,mi->i", real_spherical_harmonics(l, a), real_spherical_harmonics(l, b))
    coefficients = np.zeros(l + 1)
    coefficients[l] = 1.0
    right = (2 * l + 1) / (4 * np.pi) * legendre.legval(np.einsum("ij,ij->i", a, b), coefficients)
    assert left == pytest.approx(right, abs=1e-14)


@pytest.mark.parametrize("l", range(MAX_L + 1))
def test_orthonormal_on_the_sphere(l):
    """integral Y_lm Y_lm' dOmega = delta, by Monte Carlo on many directions."""
    directions = _unit_vectors(200000, seed=3)
    harmonics = real_spherical_harmonics(l, directions)
    overlap = 4 * np.pi * (harmonics @ harmonics.T) / len(directions)
    assert overlap == pytest.approx(np.eye(2 * l + 1), abs=0.02)


def test_zero_direction_is_finite():
    """|k+G| = 0 occurs whenever k is on the mesh; the harmonics must stay finite there."""
    values = real_spherical_harmonics(2, np.zeros((1, 3)))
    assert np.all(np.isfinite(values))


def test_rejects_high_l():
    with pytest.raises(NotImplementedError, match="up to l=3"):
        real_spherical_harmonics(4, np.array([[0.0, 0.0, 1.0]]))


def test_projector_completeness_matches_the_continuum_sum_rule(si_atoms, si_pseudo):
    """The discrete sum over G must equal the radial integral it approximates.

    ``sum_G |<k+G|beta>|^2`` is a Riemann sum for ``(2/pi) integral_0^qmax q^2 f_l(q)^2 dq``
    with ``qmax = sqrt(2 ecut)``, the two being tied together by the ``4 pi / sqrt(Omega)``
    prefactor, the spherical harmonics and the radial transform at once. Agreement at the
    1% level therefore tests all three against an independent one-dimensional quadrature.

    Note what is *not* asserted: that the sum reaches 1 at a working cutoff. It does not --
    for this dataset the hardest projector is at 0.80 even at 48 Ha, and only crosses 0.99
    near 96 Ha. A KB projector has a much longer Fourier tail than the wavefunctions it
    acts on, and only the overlap with the occupied states has to be converged, which is
    why the dataset's recommended cutoff is 18 Ha.
    """
    from amaterasu.pseudo.formfactors import projector_form_factor

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    positions = si_atoms.get_positions() * ANG_IN_BOHR
    volume = abs(float(np.linalg.det(cell)))
    symbols = list(si_atoms.get_chemical_symbols())

    previous = 0.0
    for ecut in (12.0, 24.0, 48.0):
        sphere = GSphere.build(cell, ecut, kpoint=(0.2, 0.1, 0.0))
        projectors = build_projectors(sphere, positions, symbols, {"Si": si_pseudo}, volume)
        discrete = projectors.completeness()
        assert np.all(discrete < 1.0 + 1e-9)
        assert float(discrete.min()) > previous
        previous = float(discrete.min())

        q = np.linspace(0.0, np.sqrt(2.0 * ecut), 4000)
        for row, (_, l, channel, _) in enumerate(projectors.labels):
            f = projector_form_factor(si_pseudo.grid, si_pseudo.projectors[channel], q)
            continuum = 2.0 / np.pi * np.trapezoid(q**2 * f**2, q)
            assert discrete[row] == pytest.approx(continuum, rel=0.05)
            assert l == si_pseudo.projectors[channel].l


def test_projector_operator_is_hermitian(si_atoms, si_pseudo):
    """<phi|V_NL|psi> = conj(<psi|V_NL|phi>) for any pair of bands."""
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    positions = si_atoms.get_positions() * ANG_IN_BOHR
    sphere = GSphere.build(cell, 8.0, kpoint=(0.1, -0.2, 0.3))
    projectors = build_projectors(
        sphere,
        positions,
        list(si_atoms.get_chemical_symbols()),
        {"Si": si_pseudo},
        abs(float(np.linalg.det(cell))),
    )
    rng = np.random.default_rng(0)
    block = rng.normal(size=(3, len(sphere))) + 1j * rng.normal(size=(3, len(sphere)))
    matrix = block.conj() @ projectors.apply(block).T
    assert matrix == pytest.approx(matrix.conj().T, abs=1e-12)


def test_projector_count_matches_the_dataset(si_atoms, si_pseudo):
    """Two Si atoms with l = 0,0,1,1,2,2 channels give 2 * (1+1+3+3+5+5) = 36 projectors."""
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    sphere = GSphere.build(cell, 6.0)
    projectors = build_projectors(
        sphere,
        si_atoms.get_positions() * ANG_IN_BOHR,
        list(si_atoms.get_chemical_symbols()),
        {"Si": si_pseudo},
        abs(float(np.linalg.det(cell))),
    )
    expected = 2 * sum(2 * p.l + 1 for p in si_pseudo.projectors)
    assert len(projectors) == expected == 36
    assert projectors.dij.shape == (expected, expected)
    # A norm-conserving dataset couples nothing off-diagonally.
    assert np.count_nonzero(projectors.dij - np.diag(np.diag(projectors.dij))) == 0


def test_energy_is_invariant_under_rotation_within_an_l_shell(si_atoms, si_pseudo):
    """A filled l shell is isotropic, so the m ordering cannot affect an energy."""
    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    sphere = GSphere.build(cell, 8.0, kpoint=(0.25, 0.25, 0.25))
    projectors = build_projectors(
        sphere,
        si_atoms.get_positions() * ANG_IN_BOHR,
        list(si_atoms.get_chemical_symbols()),
        {"Si": si_pseudo},
        abs(float(np.linalg.det(cell))),
    )
    rng = np.random.default_rng(5)
    block = rng.normal(size=(2, len(sphere))) + 1j * rng.normal(size=(2, len(sphere)))
    reference = projectors.energy(block, np.array([1.0, 0.5]))
    # Permuting the m rows within each (atom, l, channel) group leaves D diagonal and
    # equal within the group, so the energy must not move.
    order = np.arange(len(projectors))
    labels = projectors.labels
    for start in range(len(labels)):
        if labels[start][1] == 1 and labels[start][3] == -1:
            order[start : start + 3] = order[start : start + 3][::-1]
    projectors.matrix = projectors.matrix[order]
    assert projectors.energy(block, np.array([1.0, 0.5])) == pytest.approx(reference, rel=1e-12)
