"""Phase 1 gates for the exchange-correlation functionals.

Gate 1  agreement with libxc to 1e-10, pointwise, in energy and potential
Gate 2  the spin-polarised path reduces to the unpolarised one at zeta = 0
Gate 3  agreement with the closed-form uniform electron gas result

The finite-difference checks are the primary correctness test and need no libxc: an
analytic potential that is not the derivative of its own energy is wrong regardless of
what any reference says.
"""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.xc import _libxc_ref as ref
from amaterasu.xc import gga, lda

requires_libxc = pytest.mark.skipif(not ref.available(), reason="libxc is not installed")

#: Densities and polarisations spanning vacuum-adjacent to core-like, avoiding |zeta| = 1
#: exactly, where libxc clamps zeta at DBL_EPSILON and amaterasu takes the exact limit.
DENSITIES = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])
POLARIZATIONS = np.array([0.0, 0.3, 0.75, 0.99])
REDUCED_GRADIENTS = np.array([0.0, 0.05, 0.3, 1.0, 3.0, 10.0])


def spin_density(n, zeta):
    return np.stack([n * (1 + zeta) / 2, n * (1 - zeta) / 2], axis=-1)


def sigma_from_reduced_gradient(n, zeta, s):
    """sigma components for a total density n at polarisation zeta and reduced gradient s.

    The gradient is shared between the channels in proportion to their density, which is
    the collinear situation a real calculation produces.
    """
    k_f = (3 * np.pi**2 * n) ** (1 / 3)
    grad_total = s * 2 * k_f * n
    up = grad_total * (1 + zeta) / 2
    down = grad_total * (1 - zeta) / 2
    return np.stack([up * up, up * down, down * down], axis=-1)


@pytest.fixture(scope="module")
def grid_points():
    n, z, s = (a.ravel() for a in np.meshgrid(DENSITIES, POLARIZATIONS, REDUCED_GRADIENTS))
    return n, z, spin_density(n, z), sigma_from_reduced_gradient(n, z, s)


# ------------------------------------------------------------------ Gate 3: exact forms
def test_lda_exchange_is_exact():
    """eps_x = -(3/4)(3/pi)^(1/3) n^(1/3) for the unpolarised gas, exactly."""
    n = DENSITIES
    eps, vrho = lda.exchange(spin_density(n, np.zeros_like(n)))
    expected_eps = -0.75 * (3.0 / np.pi) ** (1 / 3) * np.cbrt(n)
    assert eps == pytest.approx(expected_eps, rel=1e-14)
    assert vrho[:, 0] == pytest.approx(4 / 3 * expected_eps, rel=1e-14)


def test_lda_exchange_spin_scaling():
    """E_x[n_up, n_dn] = (E_x[2 n_up] + E_x[2 n_dn])/2 is exact for exchange."""
    rho = np.array([[0.7, 0.3]])
    eps, _ = lda.exchange(rho)
    energy = eps[0] * 1.0

    # E_x[2 n_sigma] is the *spin-compensated* system at total density 2 n_sigma, not the
    # fully polarised one -- getting that backwards is the classic way to lose a 2^(1/3).
    doubled_up, _ = lda.exchange(np.array([[0.7, 0.7]]))
    doubled_down, _ = lda.exchange(np.array([[0.3, 0.3]]))
    assert energy == pytest.approx(0.5 * (doubled_up[0] * 1.4 + doubled_down[0] * 0.6), rel=1e-14)


def test_fully_polarized_exchange_matches_half_the_unpolarized_form():
    """At zeta = 1 only one channel contributes: eps_x = -X_FACTOR_C n^(1/3)."""
    n = 1.0
    eps, _ = lda.exchange(np.array([[n, 0.0]]))
    assert eps[0] == pytest.approx(-lda.X_FACTOR_C * n ** (1 / 3), rel=1e-15)


def test_pw92_collapses_to_its_own_branches():
    """At zeta = 0 and zeta = 1 the interpolation must return G0 and G1 unchanged."""
    rs = lda.RS_FACTOR / np.cbrt(np.array([0.1, 1.0, 10.0]))
    g0, _ = lda._pw92_g(rs, 0, lda.PW92_PARAMETERS)
    g1, _ = lda._pw92_g(rs, 1, lda.PW92_PARAMETERS)
    n = np.array([0.1, 1.0, 10.0])

    unpolarized, _ = lda.correlation_pw92(spin_density(n, np.zeros_like(n)))
    assert unpolarized == pytest.approx(g0, rel=1e-14)

    polarized, _ = lda.correlation_pw92(spin_density(n, np.ones_like(n)))
    assert polarized == pytest.approx(g1, rel=1e-14)


@pytest.mark.parametrize("functional", ["PBE", "PBEsol"])
def test_gga_exchange_reduces_to_lda_at_zero_gradient(functional):
    """F_x(0) = 1 is built into the PBE form."""
    n = DENSITIES
    rho = spin_density(n, np.zeros_like(n))
    zero_sigma = np.zeros((len(n), 3))
    eps_gga, vrho_gga, vsigma = gga.exchange(rho, zero_sigma, functional)
    eps_lda, vrho_lda = lda.exchange(rho)
    assert eps_gga == pytest.approx(eps_lda, rel=1e-14)
    assert vrho_gga == pytest.approx(vrho_lda, rel=1e-14)
    # At s = 0 the derivative dF/ds vanishes, but d/dsigma does not: it is the
    # coefficient of the leading mu*s^2 term.
    assert np.all(vsigma[:, 0] < 0.0)


@pytest.mark.parametrize("functional", ["PBE", "PBEsol"])
def test_gga_correlation_reduces_to_pw92_modified_at_zero_gradient(functional):
    """The local part of PBE correlation is the *modified* PW92, not the vanilla one."""
    n = DENSITIES
    rho = spin_density(n, np.full_like(n, 0.3))
    zero_sigma = np.zeros((len(n), 3))
    eps_gga, _, _ = gga.correlation(rho, zero_sigma, functional)
    eps_modified, _ = lda.correlation_pw92(rho, modified=True)
    eps_vanilla, _ = lda.correlation_pw92(rho, modified=False)
    assert eps_gga == pytest.approx(eps_modified, rel=1e-14)
    assert not np.allclose(eps_gga, eps_vanilla, rtol=1e-12)


def test_pbesol_is_softer_than_pbe():
    """PBEsol's smaller mu means less exchange enhancement at every finite gradient."""
    n = np.full(5, 1.0)
    zeta = np.zeros(5)
    rho = spin_density(n, zeta)
    for s in (0.1, 0.5, 1.0, 2.0, 5.0):
        sigma = sigma_from_reduced_gradient(n, zeta, np.full(5, s))
        eps_pbe, _, _ = gga.exchange(rho, sigma, "PBE")
        eps_sol, _, _ = gga.exchange(rho, sigma, "PBEsol")
        # More negative means more enhancement; PBEsol must sit between LDA and PBE.
        eps_lda, _ = lda.exchange(rho)
        assert np.all(eps_pbe < eps_sol)
        assert np.all(eps_sol < eps_lda)


def test_parameters_match_the_published_values():
    assert gga.PBE_PARAMETERS["mu"] == pytest.approx(
        gga.PBE_PARAMETERS["beta"] * np.pi**2 / 3, rel=1e-15
    )
    assert gga.PBESOL_PARAMETERS["mu"] == pytest.approx(10 / 81, rel=1e-15)
    assert gga.PBESOL_PARAMETERS["beta"] == 0.046
    assert gga.PBE_PARAMETERS["kappa"] == gga.PBESOL_PARAMETERS["kappa"] == 0.804
    # gamma = (1 - ln 2)/pi^2, the same constant as the modified PW92 A_0.
    assert gga.PBE_PARAMETERS["gamma"] == pytest.approx((1 - np.log(2)) / np.pi**2, rel=1e-15)
    assert lda.PW92_MODIFIED_PARAMETERS["a"][0] == pytest.approx(
        gga.PBE_PARAMETERS["gamma"], rel=1e-6
    )


# ------------------------------------------------------------------ Gate 2: spin limits
@pytest.mark.parametrize(
    "functional",
    [
        lda.exchange,
        lda.correlation_pw92,
        lda.correlation_pz81,
    ],
)
def test_lda_unpolarized_limit_is_symmetric(functional):
    """At zeta = 0 the two spin potentials must be identical, not merely close."""
    n = DENSITIES
    _, vrho = functional(spin_density(n, np.zeros_like(n)))
    assert vrho[:, 0] == pytest.approx(vrho[:, 1], rel=1e-15)


@pytest.mark.parametrize("functional", ["PBE", "PBEsol"])
def test_gga_unpolarized_limit_is_symmetric(functional):
    n = DENSITIES
    zeta = np.zeros_like(n)
    rho = spin_density(n, zeta)
    sigma = sigma_from_reduced_gradient(n, zeta, np.full_like(n, 0.7))
    for evaluate in (gga.exchange, gga.correlation):
        _, vrho, vsigma = evaluate(rho, sigma, functional)
        assert vrho[:, 0] == pytest.approx(vrho[:, 1], rel=1e-14)
        assert vsigma[:, 0] == pytest.approx(vsigma[:, 2], rel=1e-14)


def test_spin_flip_symmetry():
    """Swapping the two spin channels must swap the potentials and leave the energy alone."""
    rho = np.array([[0.8, 0.2]])
    sigma = np.array([[0.05, 0.02, 0.01]])
    flipped_rho = rho[:, ::-1]
    flipped_sigma = sigma[:, ::-1]
    for evaluate in (gga.exchange, gga.correlation):
        eps, vrho, vsigma = evaluate(rho, sigma, "PBE")
        eps_f, vrho_f, vsigma_f = evaluate(flipped_rho, flipped_sigma, "PBE")
        assert eps_f == pytest.approx(eps, rel=1e-14)
        assert vrho_f[:, ::-1] == pytest.approx(vrho, rel=1e-14)
        assert vsigma_f[:, ::-1] == pytest.approx(vsigma, rel=1e-14)


# --------------------------------------------------- primary check: derivative identity
def _energy(evaluate, rho, sigma):
    result = evaluate(rho, sigma) if sigma is not None else evaluate(rho)
    return result[0][0] * (rho[0, 0] + rho[0, 1])


@pytest.mark.parametrize(
    ("label", "evaluate", "needs_sigma"),
    [
        ("lda_x", lda.exchange, False),
        ("lda_c_pw", lda.correlation_pw92, False),
        ("lda_c_pz", lda.correlation_pz81, False),
        ("pbe_x", lambda r, s: gga.exchange(r, s, "PBE"), True),
        ("pbe_c", lambda r, s: gga.correlation(r, s, "PBE"), True),
        ("pbesol_x", lambda r, s: gga.exchange(r, s, "PBEsol"), True),
        ("pbesol_c", lambda r, s: gga.correlation(r, s, "PBEsol"), True),
    ],
)
@pytest.mark.parametrize(
    ("n", "zeta", "s"), [(1.0, 0.0, 0.5), (1.0, 0.4, 1.5), (0.1, 0.8, 0.2), (10.0, 0.2, 3.0)]
)
def test_potential_is_the_derivative_of_the_energy(label, evaluate, needs_sigma, n, zeta, s):
    """v = d(n eps)/dn by central differences. This is the test that catches sign errors."""
    rho = spin_density(np.array([n]), np.array([zeta]))
    sigma = sigma_from_reduced_gradient(np.array([n]), np.array([zeta]), np.array([s]))
    call = (lambda r, sg: evaluate(r, sg)) if needs_sigma else (lambda r, sg: evaluate(r))

    _, vrho, *rest = call(rho, sigma)
    for spin in range(2):
        step = 1e-6 * max(rho[0, spin], n * 1e-3)
        plus, minus = rho.copy(), rho.copy()
        plus[0, spin] += step
        minus[0, spin] -= step
        finite = (_energy_of(call, plus, sigma) - _energy_of(call, minus, sigma)) / (2 * step)
        assert vrho[0, spin] == pytest.approx(finite, rel=2e-6, abs=1e-9)

    if rest:
        vsigma = rest[0]
        for component in range(3):
            scale = max(abs(sigma[0, component]), sigma.max() * 1e-3, 1e-12)
            step = 1e-6 * scale
            plus, minus = sigma.copy(), sigma.copy()
            plus[0, component] += step
            minus[0, component] -= step
            finite = (_energy_of(call, rho, plus) - _energy_of(call, rho, minus)) / (2 * step)
            assert vsigma[0, component] == pytest.approx(finite, rel=1e-5, abs=1e-10)


def _energy_of(call, rho, sigma):
    eps = call(rho, sigma)[0]
    return eps[0] * (rho[0, 0] + rho[0, 1])


# ------------------------------------------------------- vacuum and threshold behaviour
@pytest.mark.parametrize(
    "evaluate",
    [
        lambda r, s: lda.exchange(r),
        lambda r, s: lda.correlation_pw92(r),
        lambda r, s: lda.correlation_pz81(r),
        lambda r, s: gga.exchange(r, s, "PBE"),
        lambda r, s: gga.correlation(r, s, "PBEsol"),
    ],
)
def test_vacuum_is_finite_and_zero(evaluate):
    """A real density has vacuum regions; the functional must not produce NaN there."""
    rho = np.array([[0.0, 0.0], [1e-30, 0.0], [1e-30, 1e-30]])
    sigma = np.zeros((3, 3))
    outputs = evaluate(rho, sigma)
    for array in outputs:
        assert np.all(np.isfinite(array)), "vacuum produced a non-finite value"
    assert outputs[0] == pytest.approx(np.zeros(3), abs=1e-30)


# ------------------------------------------------------------------ Gate 1: vs libxc
@requires_libxc
@pytest.mark.parametrize(
    ("label", "evaluate", "reference"),
    [
        ("lda_x", lambda r, s: lda.exchange(r), "lda_x"),
        ("lda_c_pw", lambda r, s: lda.correlation_pw92(r), "lda_c_pw"),
        ("lda_c_pw_mod", lambda r, s: lda.correlation_pw92(r, modified=True), "lda_c_pw_mod"),
        ("lda_c_pz", lambda r, s: lda.correlation_pz81(r), "lda_c_pz"),
        ("gga_x_pbe", lambda r, s: gga.exchange(r, s, "PBE"), "gga_x_pbe"),
        ("gga_x_pbe_sol", lambda r, s: gga.exchange(r, s, "PBEsol"), "gga_x_pbe_sol"),
        ("gga_c_pbe", lambda r, s: gga.correlation(r, s, "PBE"), "gga_c_pbe"),
        ("gga_c_pbe_sol", lambda r, s: gga.correlation(r, s, "PBEsol"), "gga_c_pbe_sol"),
    ],
)
def test_matches_libxc(label, evaluate, reference, grid_points):
    """Gate 1: pointwise agreement to 1e-10 in energy, vrho and vsigma."""
    _, _, rho, sigma = grid_points
    is_gga = reference.startswith("gga")
    mine = evaluate(rho, sigma)

    kwargs = {"spin": "polarized"}
    args = (rho.ravel(), sigma.ravel()) if is_gga else (rho.ravel(), None)
    eps_ref = ref.evaluate(reference, *args, **kwargs)
    vrho_ref = ref.evaluate(reference, *args, key="vrho", **kwargs).reshape(-1, 2)

    assert mine[0] == pytest.approx(eps_ref, rel=1e-10, abs=1e-16)
    assert mine[1] == pytest.approx(vrho_ref, rel=1e-10, abs=1e-16)
    if is_gga:
        vsigma_ref = ref.evaluate(reference, *args, key="vsigma", **kwargs).reshape(-1, 3)
        assert mine[2] == pytest.approx(vsigma_ref, rel=1e-10, abs=1e-16)
