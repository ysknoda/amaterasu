"""Smearing schemes, the Fermi level and the entropy term."""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.scf.occupations import (
    compute_occupations,
    entropy_function,
    occupation_function,
)

SYMMETRIC = ["fermi-dirac", "gaussian", "methfessel-paxton"]
ALL = [*SYMMETRIC, "marzari-vanderbilt"]


@pytest.mark.parametrize("scheme", ALL)
def test_limits(scheme):
    """Deep below the Fermi level a state is full, far above it is empty."""
    assert occupation_function(np.array([-40.0]), scheme, 1)[0] == pytest.approx(1.0, abs=1e-12)
    assert occupation_function(np.array([40.0]), scheme, 1)[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("scheme", SYMMETRIC)
def test_particle_hole_symmetry(scheme):
    """f(x) + f(-x) = 1 for every scheme but cold smearing, which is deliberately skewed."""
    x = np.linspace(-8.0, 8.0, 101)
    assert occupation_function(x, scheme, 1) + occupation_function(-x, scheme, 1) == pytest.approx(
        np.ones_like(x), abs=1e-14
    )


def test_cold_smearing_is_asymmetric_by_design():
    """Marzari-Vanderbilt trades symmetry for a free energy that tracks E to third order."""
    assert occupation_function(np.array([0.0]), "marzari-vanderbilt")[0] != pytest.approx(0.5)


def test_known_entropy_values_at_the_fermi_level():
    """S(0) is a closed-form number for each scheme, so it pins the normalisation."""
    assert entropy_function(np.array([0.0]), "fermi-dirac")[0] == pytest.approx(np.log(2.0))
    assert entropy_function(np.array([0.0]), "gaussian")[0] == pytest.approx(0.5 / np.sqrt(np.pi))
    # Order-1 Methfessel-Paxton: S = (1/2) A_1 H_2(0) exp(0) with A_1 = -1/(4 sqrt(pi)),
    # H_2(0) = -2, giving 1/(4 sqrt(pi)).
    assert entropy_function(np.array([0.0]), "methfessel-paxton", 1)[0] == pytest.approx(
        0.25 / np.sqrt(np.pi)
    )


@pytest.mark.parametrize("scheme", ALL)
def test_entropy_vanishes_far_from_the_fermi_level(scheme):
    """Only partially filled states carry entropy.

    The tolerance is set by Fermi-Dirac, whose tails are exponential rather than
    Gaussian: at x = 30 the occupation is exp(-30) = 9e-14 and the entropy
    -f (ln f - 1) is 3e-12, whereas every other scheme here is already at zero to
    machine precision.
    """
    assert entropy_function(np.array([-30.0, 30.0]), scheme, 1) == pytest.approx(
        np.zeros(2), abs=1e-10
    )


def test_gaussian_entropy_is_the_derivative_of_the_occupation():
    """For Gaussian smearing, S(x) = -x f'(x)/... check the underlying delta instead:
    df/dx = -exp(-x^2)/sqrt(pi), which is -2 S(x)."""
    x = np.linspace(-3.0, 3.0, 61)
    h = 1e-5
    derivative = (
        occupation_function(x + h, "gaussian") - occupation_function(x - h, "gaussian")
    ) / (2 * h)
    assert derivative == pytest.approx(-2.0 * entropy_function(x, "gaussian"), rel=1e-6)


@pytest.mark.parametrize("scheme", ALL)
def test_electron_count_is_exact(scheme):
    """The Fermi level is defined by the electron count, so that must come back exactly."""
    rng = np.random.default_rng(0)
    eigenvalues = np.sort(rng.normal(size=(1, 6, 12)), axis=-1)
    weights = np.full(6, 1.0 / 6.0)
    result = compute_occupations(eigenvalues, weights, 9.0, scheme=scheme, smearing=0.02, nspin=1)
    count = 2.0 * float(np.einsum("k,skn->", weights, result.occupations))
    assert count == pytest.approx(9.0, abs=1e-9)


def test_spin_polarised_counting_and_moment():
    eigenvalues = np.zeros((2, 2, 4))
    eigenvalues[0] = np.array([-1.0, -0.9, 0.5, 0.6])  # majority sits lower
    eigenvalues[1] = np.array([-0.5, -0.4, 0.9, 1.0])
    weights = np.array([0.5, 0.5])
    result = compute_occupations(
        eigenvalues, weights, 4.0, scheme="gaussian", smearing=0.05, nspin=2
    )
    count = float(np.einsum("k,skn->", weights, result.occupations))
    assert count == pytest.approx(4.0, abs=1e-9)
    assert result.magnetization == pytest.approx(0.0, abs=1e-9)  # both channels fill 2 states
    assert result.occupations.shape == (2, 2, 4)


def test_fixed_occupations_fill_from_the_bottom_and_find_the_gap():
    eigenvalues = np.array([[[-2.0, -1.0, 3.0, 4.0]]])
    result = compute_occupations(
        np.tile(eigenvalues, (1, 3, 1)), np.full(3, 1 / 3), 4.0, scheme="fixed", nspin=1
    )
    assert result.occupations[0, 0] == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert result.gap == pytest.approx(4.0)
    assert result.fermi_level == pytest.approx(1.0)
    assert result.entropy_term == 0.0
    assert not result.is_metallic


def test_gap_is_detected_under_smearing():
    """An insulator is an insulator whether or not smearing was switched on."""
    eigenvalues = np.tile(np.array([[[-2.0, -1.9, 3.0, 3.1]]]), (1, 2, 1))
    result = compute_occupations(
        eigenvalues, np.full(2, 0.5), 4.0, scheme="gaussian", smearing=0.05, nspin=1
    )
    assert result.gap == pytest.approx(4.9, rel=1e-6)


def test_smearing_lowers_the_free_energy():
    """F = E - sigma S with S >= 0, so the entropy term must be non-negative here."""
    rng = np.random.default_rng(3)
    eigenvalues = np.sort(rng.normal(size=(1, 4, 8)), axis=-1)
    result = compute_occupations(
        eigenvalues, np.full(4, 0.25), 6.0, scheme="gaussian", smearing=0.1, nspin=1
    )
    assert result.entropy_term > 0.0


def test_too_few_bands_is_reported_clearly():
    eigenvalues = np.zeros((1, 1, 2))
    with pytest.raises(ValueError, match=r"raise occupations\.nbands"):
        compute_occupations(eigenvalues, np.ones(1), 10.0, scheme="gaussian", smearing=0.1)


def test_tetrahedron_is_deferred_with_a_useful_message():
    with pytest.raises(NotImplementedError, match="Phase 5"):
        compute_occupations(
            np.zeros((1, 1, 2)), np.ones(1), 2.0, scheme="tetrahedron", smearing=0.1
        )


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown smearing scheme"):
        occupation_function(np.array([0.0]), "made-up")
