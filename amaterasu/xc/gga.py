"""PBE and PBEsol: the two generalised gradient approximations amaterasu supports.

PBE and PBEsol share one functional form and differ only in two numbers. PBEsol restores
the second-order gradient expansion for exchange (``mu = 10/81`` instead of ``0.21951``)
and weakens the correlation gradient term (``beta = 0.046`` instead of ``0.066725``),
which is why it gives better lattice constants for densely packed solids while PBE is
better for molecular atomisation energies.

    | functional | mu                  | beta        | kappa |
    |------------|---------------------|-------------|-------|
    | PBE        | 0.2195149727645171  | 0.066724551 | 0.804 |
    | PBEsol     | 10/81 = 0.123456790 | 0.046       | 0.804 |

Every one of these was confirmed numerically against libxc 7.1.2 rather than transcribed
from the papers; see ``tests/unit/test_libxc_reference.py``.

Numerically stable forms
------------------------
Two rewrites, both taken from libxc's own derivation, matter at small gradients where a
naive implementation loses most of its significant digits:

* the enhancement factor is evaluated as ``F_x = 1 + kappa mu s^2/(kappa + mu s^2)``
  rather than ``1 + kappa - kappa/(1 + mu s^2/kappa)``, which would compute
  ``1 - (something very close to 1)``;
* the correlation uses ``expm1`` and ``log1p`` instead of ``exp(x) - 1`` and
  ``log(1 + x)``.

Correlation is built on the **modified** PW92 parametrisation. That is not a detail:
building it on the vanilla one leaves a systematic 2.5e-7 Ha/electron offset, since the
modified A_0 = 0.0310907 is the same constant as PBE's ``gamma = (1 - ln 2)/pi^2``.

Conventions follow :mod:`amaterasu.xc.lda`: densities are ``(npoints, 2)``, gradients are
passed as libxc's contracted ``sigma = (grad n_up . grad n_up, grad n_up . grad n_dn,
grad n_dn . grad n_dn)`` of shape ``(npoints, 3)``, and the returned ``vsigma`` has the
same layout.
"""

from __future__ import annotations

import numpy as np

from .lda import (
    DENSITY_THRESHOLD,
    X_FACTOR_C,
    _pw92_g,
    f_zeta,
    f_zeta_derivative,
    rs_and_zeta,
)

__all__ = [
    "PBESOL_PARAMETERS",
    "PBE_PARAMETERS",
    "X2S",
    "correlation",
    "exchange",
    "phi_zeta",
]

#: 1/(2 (6 pi^2)^(1/3)); converts the per-spin reduced gradient x = |grad n_s|/n_s^(4/3)
#: into the spin-scaled s that the enhancement factor takes.
X2S = 0.1282782438530421943003109254455883701296

#: Exchange kappa and mu, and correlation beta and gamma. gamma = (1 - ln 2)/pi^2.
PBE_PARAMETERS = {
    "kappa": 0.8040,
    "mu": 0.2195149727645171,
    "beta": 0.06672455060314922,
    "gamma": 0.031090690869654895034,
}
PBESOL_PARAMETERS = {
    "kappa": 0.8040,
    "mu": 0.1234567901234567901234567901234567901235,  # 10/81
    "beta": 0.046,
    "gamma": 0.031090690869654895034,
}

_SIGMA_FLOOR = 1e-40


def _resolve(functional: str) -> dict:
    key = functional.strip().lower().replace("-", "").replace("_", "")
    if key in ("pbe",):
        return PBE_PARAMETERS
    if key in ("pbesol",):
        return PBESOL_PARAMETERS
    raise ValueError(f"unknown GGA functional {functional!r}; expected 'PBE' or 'PBEsol'")


def _as_arrays(rho, sigma):
    rho = np.atleast_2d(np.asarray(rho, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    if rho.shape[-1] != 2:
        raise ValueError(f"spin density must have shape (npoints, 2), got {rho.shape}")
    if sigma.shape[-1] != 3:
        raise ValueError(f"sigma must have shape (npoints, 3), got {sigma.shape}")
    if rho.shape[0] != sigma.shape[0]:
        raise ValueError(f"rho has {rho.shape[0]} points but sigma has {sigma.shape[0]}")
    return rho, sigma


def phi_zeta(zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spin-scaling factor ``phi = [(1+z)^(2/3) + (1-z)^(2/3)]/2`` and ``dphi/dz``."""
    opz = np.maximum(1.0 + zeta, DENSITY_THRESHOLD)
    omz = np.maximum(1.0 - zeta, DENSITY_THRESHOLD)
    phi = 0.5 * (opz ** (2 / 3) + omz ** (2 / 3))
    dphi = (1 / 3) * (opz ** (-1 / 3) - omz ** (-1 / 3))
    return phi, dphi


# --------------------------------------------------------------------------- exchange
def exchange(rho, sigma, functional: str = "PBE"):
    """PBE / PBEsol exchange.

    Exchange obeys the exact spin-scaling relation, so the two spin channels are
    independent and there is no cross term: ``vsigma`` has a zero up-down component.

        e_x = -X_FACTOR_C sum_s n_s^(4/3) F_x(s_s),   s_s = X2S |grad n_s| / n_s^(4/3)
        F_x(s) = 1 + kappa mu s^2 / (kappa + mu s^2)
    """
    rho, sigma = _as_arrays(rho, sigma)
    parameters = _resolve(functional)
    kappa, mu = parameters["kappa"], parameters["mu"]

    total = rho[:, 0] + rho[:, 1]
    active = total > DENSITY_THRESHOLD

    spin = np.maximum(rho, 0.0)
    spin_sigma = np.stack([np.maximum(sigma[:, 0], 0.0), np.maximum(sigma[:, 2], 0.0)], axis=-1)
    safe_spin = np.maximum(spin, DENSITY_THRESHOLD)

    n43 = safe_spin ** (4 / 3)
    s = X2S * np.sqrt(spin_sigma) / n43
    s2 = s * s
    denominator = kappa + mu * s2

    enhancement = 1.0 + kappa * mu * s2 / denominator
    # F_x'(s) * s = 2 kappa^2 mu s^2 / (kappa + mu s^2)^2
    enhancement_times_s = 2.0 * kappa * kappa * mu * s2 / denominator**2

    channel_energy = -X_FACTOR_C * n43 * enhancement
    channel_energy = np.where(spin > DENSITY_THRESHOLD, channel_energy, 0.0)
    energy_density = channel_energy.sum(axis=1)

    eps = np.zeros_like(total)
    np.divide(energy_density, total, out=eps, where=active)

    vrho = -(4 / 3) * X_FACTOR_C * np.cbrt(safe_spin) * (enhancement - enhancement_times_s)
    # d e_x/d sigma_ss written without dividing by sigma, so sigma = 0 is safe.
    vsigma_spin = -X_FACTOR_C * kappa * kappa * mu * X2S**2 / (n43 * denominator**2)

    keep = (spin > DENSITY_THRESHOLD) & active[:, None]
    vrho = np.where(keep, vrho, 0.0)
    vsigma_spin = np.where(keep, vsigma_spin, 0.0)
    vsigma = np.stack([vsigma_spin[:, 0], np.zeros_like(total), vsigma_spin[:, 1]], axis=-1)
    return np.where(active, eps, 0.0), vrho, vsigma


# ------------------------------------------------------------------------ correlation
def correlation(rho, sigma, functional: str = "PBE"):
    """PBE / PBEsol correlation, on top of the modified PW92 local part.

    eps_c = eps_c^PW92mod(rs, z) + H,
    H = gamma phi^3 log1p( beta u / (gamma (1 + A u)) ),  u = t^2 + A t^4,
    A = beta / (gamma expm1(-eps_c^LDA/(gamma phi^3))).
    """
    rho, sigma = _as_arrays(rho, sigma)
    parameters = _resolve(functional)
    beta, gamma = parameters["beta"], parameters["gamma"]

    total, rs, zeta = rs_and_zeta(rho)
    active = total > DENSITY_THRESHOLD
    safe_total = np.maximum(total, DENSITY_THRESHOLD)

    # --- local part: modified PW92, and its rs and zeta derivatives -------------
    from .lda import PW92_MODIFIED_PARAMETERS as pw

    g0, dg0 = _pw92_g(rs, 0, pw)
    g1, dg1 = _pw92_g(rs, 1, pw)
    g2, dg2 = _pw92_g(rs, 2, pw)
    fz20 = pw["fz20"]
    fz, dfz = f_zeta(zeta), f_zeta_derivative(zeta)
    z3, z4 = zeta**3, zeta**4
    bracket = g1 - g0 + g2 / fz20

    eps_lda = g0 + z4 * fz * bracket - fz * g2 / fz20
    deps_lda_drs = dg0 + z4 * fz * (dg1 - dg0 + dg2 / fz20) - fz * dg2 / fz20
    deps_lda_dz = (4.0 * z3 * fz + z4 * dfz) * bracket - dfz * g2 / fz20

    # --- gradient part ---------------------------------------------------------
    phi, dphi_dz = phi_zeta(zeta)
    phi3 = phi**3

    sigma_total = np.maximum(sigma[:, 0] + 2.0 * sigma[:, 1] + sigma[:, 2], 0.0)
    # t = xt / (4 * 2^(1/3) * phi * sqrt(rs)) with xt = sqrt(sigma_tot)/n^(4/3).
    # Keeping the prefactor separate lets t^2/sigma be formed without dividing by sigma.
    t_per_xt = 1.0 / (4.0 * 2.0 ** (1 / 3) * phi * np.sqrt(rs))
    xt = np.sqrt(sigma_total) / safe_total ** (4 / 3)
    t = t_per_xt * xt
    t2 = t * t

    expm1_argument = -eps_lda / (gamma * phi3)
    e_term = np.expm1(expm1_argument)
    # A diverges as the local correlation vanishes; clamp so the ratio stays finite.
    e_safe = np.where(np.abs(e_term) < 1e-300, 1e-300, e_term)
    a_coeff = beta / (gamma * e_safe)

    u = t2 + a_coeff * t2 * t2
    d_term = 1.0 + a_coeff * u
    w = beta * u / (gamma * d_term)
    log_term = np.log1p(w)
    h_energy = gamma * phi3 * log_term

    # --- derivatives of H with respect to its immediate arguments ---------------
    prefactor = gamma * phi3 / (1.0 + w)
    dw_du = beta / (gamma * d_term**2)
    dh_dt = prefactor * dw_du * (2.0 * t + 4.0 * a_coeff * t2 * t)
    dh_da = prefactor * beta * (t2 * t2 - u * u) / (gamma * d_term**2)
    dh_dphi_explicit = 3.0 * gamma * phi * phi * log_term

    da_deps = beta * (e_term + 1.0) / (gamma**2 * phi3 * e_safe**2)
    da_dphi = -3.0 * beta * eps_lda * (e_term + 1.0) / (gamma**2 * phi**4 * e_safe**2)

    # t depends on phi and rs explicitly, and on n through xt.
    dh_drs = dh_da * da_deps * deps_lda_drs + dh_dt * (-t / (2.0 * rs))
    dh_dz = (
        dh_dphi_explicit * dphi_dz
        + dh_da * (da_dphi * dphi_dz + da_deps * deps_lda_dz)
        + dh_dt * (-t / phi) * dphi_dz
    )
    dh_dxt = dh_dt * t_per_xt

    # --- assemble ---------------------------------------------------------------
    eps = eps_lda + h_energy
    deps_drs = deps_lda_drs + dh_drs
    deps_dz = deps_lda_dz + dh_dz

    common = eps - (rs / 3.0) * deps_drs - (4 / 3) * xt * dh_dxt
    vrho = np.stack([common + (1.0 - zeta) * deps_dz, common - (1.0 + zeta) * deps_dz], axis=-1)

    # d(n eps)/d sigma_tot = n dH/dt * dt/dsigma_tot, with dt/dsigma = t/(2 sigma).
    # t^2/sigma = t_per_xt^2 / n^(8/3) is finite at sigma = 0, so form it that way.
    t2_over_sigma = t_per_xt**2 / safe_total ** (8 / 3)
    dh_dsigma = (
        phi3 * beta * t2_over_sigma * (2.0 + 4.0 * a_coeff * t2) / (2.0 * (1.0 + w) * d_term**2)
    )
    vsigma_total = safe_total * dh_dsigma

    eps = np.where(active, eps, 0.0)
    vrho = np.where(active[:, None], vrho, 0.0)
    vsigma_total = np.where(active, vsigma_total, 0.0)
    # eps depends on sigma only through sigma_tot = sigma_uu + 2 sigma_ud + sigma_dd.
    vsigma = np.stack([vsigma_total, 2.0 * vsigma_total, vsigma_total], axis=-1)
    return eps, vrho, vsigma
