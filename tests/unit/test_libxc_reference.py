"""The libxc reference bridge: is the installed libxc the one we think it is?

These tests do not check amaterasu's own exchange-correlation code, which lands in Phase 1.
They check the *reference* against which that code will be validated, and they pin down the
PBE/PBEsol parameters numerically so Phase 1 hard-codes measured values rather than
remembered ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.xc import _libxc_ref as ref

pytestmark = pytest.mark.skipif(
    not ref.available(),
    reason="libxc is not installed; see amaterasu.xc._libxc_ref for the build recipe",
)


def lda_exchange_analytic(rho):
    """eps_x = -(3/4)(3/pi)^(1/3) n^(1/3); v_x = (4/3) eps_x. Exact, no approximation."""
    eps = -0.75 * (3.0 / np.pi) ** (1 / 3) * np.asarray(rho) ** (1 / 3)
    return eps, 4.0 / 3.0 * eps


def test_libxc_is_importable_and_reports_a_version():
    assert ref.version() is not None


def test_lda_exchange_matches_the_closed_form():
    """If libxc disagrees with the exact LDA exchange, it is not a usable reference."""
    rho = np.array([1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0])
    eps_expected, v_expected = lda_exchange_analytic(rho)
    assert ref.evaluate("lda_x", rho) == pytest.approx(eps_expected, rel=1e-13)
    assert ref.evaluate("lda_x", rho, key="vrho") == pytest.approx(v_expected, rel=1e-13)


@pytest.mark.parametrize("name", ["gga_x_pbe", "gga_x_pbe_sol"])
def test_gga_exchange_reduces_to_lda_at_zero_gradient(name):
    """F_x(0) = 1 is built into the PBE form; a deviation means a different functional."""
    rho = np.array([0.1, 1.0, 5.0])
    eps_lda, _ = lda_exchange_analytic(rho)
    assert ref.evaluate(name, rho, np.zeros_like(rho)) == pytest.approx(eps_lda, rel=1e-13)


def test_pbe_correlation_reduces_to_pw92_mod_not_pw92():
    """A subtle trap: libxc's PBE correlation uses lda_c_pw_mod at zero gradient.

    Implementing PBE on top of plain lda_c_pw instead leaves a systematic offset of about
    2.5e-7 Ha/electron -- small, but a permanent floor on any comparison against libxc.
    """
    for rho in (0.1, 1.0, 5.0):
        at_zero = ref.evaluate("gga_c_pbe", rho, 0.0)[0]
        assert at_zero == pytest.approx(ref.evaluate("lda_c_pw_mod", rho)[0], abs=1e-15)
        assert abs(at_zero - ref.evaluate("lda_c_pw", rho)[0]) > 1e-8


@pytest.mark.parametrize(
    ("name", "parameters"),
    [("gga_x_pbe", ref.PBE_PARAMETERS), ("gga_x_pbe_sol", ref.PBESOL_PARAMETERS)],
)
def test_exchange_mu(name, parameters):
    """mu = 0.21951 for PBE; PBEsol restores the gradient expansion value 10/81."""
    assert ref.extract_exchange_mu(name) == pytest.approx(parameters["mu"], rel=1e-6)


@pytest.mark.parametrize(
    ("name", "parameters"),
    [("gga_c_pbe", ref.PBE_PARAMETERS), ("gga_c_pbe_sol", ref.PBESOL_PARAMETERS)],
)
def test_correlation_beta(name, parameters):
    """beta = 0.066725 for PBE, 0.046 for PBEsol."""
    assert ref.extract_correlation_beta(name) == pytest.approx(parameters["beta"], rel=1e-6)


@pytest.mark.parametrize("name", ["gga_x_pbe", "gga_x_pbe_sol"])
def test_exchange_kappa(name):
    """F_x saturates at 1 + kappa as s grows; kappa = 0.804 is shared by PBE and PBEsol."""
    assert ref.exchange_enhancement(name, 1.0, 1e4) - 1.0 == pytest.approx(0.804, rel=1e-6)


def test_pbesol_is_softer_than_pbe():
    """PBEsol's smaller mu means less exchange enhancement at every finite gradient.

    This is the whole point of PBEsol: a weaker gradient correction that restores the
    slowly-varying-density limit and improves lattice constants of dense solids.
    """
    for s in (0.1, 0.5, 1.0, 2.0):
        pbe = ref.exchange_enhancement("gga_x_pbe", 1.0, s)
        pbesol = ref.exchange_enhancement("gga_x_pbe_sol", 1.0, s)
        assert 1.0 <= pbesol < pbe


def test_spin_polarized_reduces_to_unpolarized():
    """At zeta = 0 the two branches must agree exactly, or the spin path has a bug."""
    rho = 0.7
    sigma = ref.reduced_gradient_sigma(rho, 0.5)
    unpolarized = ref.evaluate("gga_x_pbe", rho, sigma)[0]
    polarized = ref.evaluate(
        "gga_x_pbe",
        [rho / 2, rho / 2],
        [sigma / 4, sigma / 4, sigma / 4],
        spin="polarized",
    )[0]
    assert polarized == pytest.approx(unpolarized, rel=1e-13)
