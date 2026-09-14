"""Local density approximation: exchange, and the PZ81 and PW92 correlation functionals.

Conventions
-----------
All quantities are in Hartree atomic units. Densities are passed as an array of shape
``(npoints, 2)`` holding the spin-up and spin-down densities separately; an unpolarised
calculation passes ``n/2`` in both columns. This mirrors libxc's spin-polarised layout so
the two can be compared without a convention shim in between -- a shim is exactly where a
factor of two hides.

Each functional returns

``eps``
    the exchange-correlation energy **per particle**, ``e_xc / n`` (libxc's ``zk``);
``vrho``
    ``d(n eps)/d n_sigma``, shape ``(npoints, 2)`` (libxc's ``vrho``).

Normalisation constants are taken from libxc's ``src/util.h`` so that both codes are
demonstrably parametrising the same functional:

    X_FACTOR_C = (3/8)(3/pi)^(1/3) 4^(2/3)   RS_FACTOR = (3/(4 pi))^(1/3)

PZ81 and PW92 parameters are transcribed from libxc's ``lda_c_pz.c`` and ``lda_c_pw.c``.
Note the ``modified`` flag on PW92: PBE correlation is built on the *modified*
parametrisation, whose A coefficients carry more digits (A_0 = 0.0310907 is
(1 - ln 2)/pi^2 rounded, the same constant PBE calls gamma) and whose f''(0) is exact.
Building PBE on the vanilla parametrisation instead leaves a systematic offset of about
2.5e-7 Ha/electron.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PW92_MODIFIED_PARAMETERS",
    "PW92_PARAMETERS",
    "PZ81_PARAMETERS",
    "RS_FACTOR",
    "X_FACTOR_C",
    "correlation_pw92",
    "correlation_pz81",
    "exchange",
    "f_zeta",
    "f_zeta_derivative",
    "rs_and_zeta",
]

#: (3/8)(3/pi)^(1/3) 4^(2/3); e_x = -X_FACTOR_C * sum_sigma n_sigma^(4/3).
X_FACTOR_C = 0.9305257363491000250020102180716672510262
#: (3/(4 pi))^(1/3); the Wigner-Seitz radius is RS_FACTOR / n^(1/3).
RS_FACTOR = 0.6203504908994000166680068120477781673508

#: Densities below this are treated as vacuum: the functional returns zero there rather
#: than dividing by a number that is numerically zero.
DENSITY_THRESHOLD = 1e-24

_FZ_DENOMINATOR = 2.0 ** (4.0 / 3.0) - 2.0

#: PW92 parameters, index order [unpolarised, polarised, -alpha_c]. From libxc lda_c_pw.c.
PW92_PARAMETERS = {
    "a": (0.031091, 0.015545, 0.016887),
    "alpha1": (0.21370, 0.20548, 0.11125),
    "beta1": (7.5957, 14.1189, 10.357),
    "beta2": (3.5876, 6.1977, 3.6231),
    "beta3": (1.6382, 3.3662, 0.88026),
    "beta4": (0.49294, 0.62517, 0.49671),
    "fz20": 1.709921,
}

#: PW92 "modified": more digits in A and the exact f''(0). This is what PBE is built on.
PW92_MODIFIED_PARAMETERS = {
    **PW92_PARAMETERS,
    "a": (0.0310907, 0.01554535, 0.0168869),
    "fz20": 1.709920934161365617563962776245,
}

#: PZ81 parameters as (unpolarised, polarised). From libxc lda_c_pz.c.
PZ81_PARAMETERS = {
    "gamma": (-0.1423, -0.0843),
    "beta1": (1.0529, 1.3981),
    "beta2": (0.3334, 0.2611),
    "a": (0.0311, 0.01555),
    "b": (-0.048, -0.0269),
    "c": (0.0020, 0.0007),
    "d": (-0.0116, -0.0048),
}


def _as_spin_density(rho) -> np.ndarray:
    """Coerce input to a ``(npoints, 2)`` spin density array."""
    rho = np.atleast_2d(np.asarray(rho, dtype=float))
    if rho.shape[-1] != 2:
        raise ValueError(f"spin density must have shape (npoints, 2), got {rho.shape}")
    return rho


def rs_and_zeta(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n, rs, zeta)`` for a spin density, clamped away from vacuum."""
    total = rho[:, 0] + rho[:, 1]
    safe = np.maximum(total, DENSITY_THRESHOLD)
    rs = RS_FACTOR / np.cbrt(safe)
    zeta = np.clip((rho[:, 0] - rho[:, 1]) / safe, -1.0, 1.0)
    return total, rs, zeta


def f_zeta(zeta: np.ndarray) -> np.ndarray:
    """Spin interpolation ``f(z) = [(1+z)^(4/3) + (1-z)^(4/3) - 2] / (2^(4/3) - 2)``."""
    opz = np.maximum(1.0 + zeta, 0.0)
    omz = np.maximum(1.0 - zeta, 0.0)
    return (opz ** (4 / 3) + omz ** (4 / 3) - 2.0) / _FZ_DENOMINATOR


def f_zeta_derivative(zeta: np.ndarray) -> np.ndarray:
    """``df/dz``, which vanishes at ``z = 0`` and diverges in slope at ``|z| = 1``."""
    opz = np.maximum(1.0 + zeta, 0.0)
    omz = np.maximum(1.0 - zeta, 0.0)
    return (4 / 3) * (np.cbrt(opz) - np.cbrt(omz)) / _FZ_DENOMINATOR


# --------------------------------------------------------------------------- exchange
def exchange(rho) -> tuple[np.ndarray, np.ndarray]:
    """LDA exchange.

    The exact spin-scaling relation ``E_x[n_up, n_dn] = (E_x[2 n_up] + E_x[2 n_dn]) / 2``
    collapses to a sum over spins with no interpolation function:

        e_x = -X_FACTOR_C sum_sigma n_sigma^(4/3),
        v_x^sigma = -(4/3) X_FACTOR_C n_sigma^(1/3).
    """
    rho = _as_spin_density(rho)
    total = rho[:, 0] + rho[:, 1]
    active = total > DENSITY_THRESHOLD

    spin = np.maximum(rho, 0.0)
    cube_root = np.cbrt(spin)
    energy_density = -X_FACTOR_C * np.sum(cube_root * spin, axis=1)

    eps = np.zeros_like(total)
    np.divide(energy_density, total, out=eps, where=active)
    vrho = np.where(active[:, None], -(4 / 3) * X_FACTOR_C * cube_root, 0.0)
    return eps, vrho


# ------------------------------------------------------------------------- PW92
def _pw92_g(rs: np.ndarray, index: int, parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    """PW92 equation (10): the function ``G(rs)`` and its derivative ``dG/drs``.

    ``log1p`` rather than ``log(1 + x)`` because the argument is small at large rs, where
    the direct form loses most of its significant digits.
    """
    a = parameters["a"][index]
    alpha1 = parameters["alpha1"][index]
    b1, b2 = parameters["beta1"][index], parameters["beta2"][index]
    b3, b4 = parameters["beta3"][index], parameters["beta4"][index]

    sqrt_rs = np.sqrt(rs)
    q = b1 * sqrt_rs + b2 * rs + b3 * rs * sqrt_rs + b4 * rs * rs
    dq = 0.5 * b1 / sqrt_rs + b2 + 1.5 * b3 * sqrt_rs + 2.0 * b4 * rs

    inner = 1.0 / (2.0 * a * q)
    log_term = np.log1p(inner)
    g = -2.0 * a * (1.0 + alpha1 * rs) * log_term
    # d/drs log1p(1/(2 a Q)) = -dQ / (2 a Q^2 + Q)
    dg = -2.0 * a * alpha1 * log_term + 2.0 * a * (1.0 + alpha1 * rs) * dq / (2.0 * a * q * q + q)
    return g, dg


def correlation_pw92(rho, *, modified: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Perdew-Wang 1992 correlation (PRB 45, 13244).

    Parameters
    ----------
    modified
        Use the parametrisation with extra digits in A and the exact ``f''(0)``. PBE
        correlation is defined on top of this variant, not the vanilla one.
    """
    rho = _as_spin_density(rho)
    parameters = PW92_MODIFIED_PARAMETERS if modified else PW92_PARAMETERS
    total, rs, zeta = rs_and_zeta(rho)
    active = total > DENSITY_THRESHOLD

    g0, dg0 = _pw92_g(rs, 0, parameters)  # unpolarised
    g1, dg1 = _pw92_g(rs, 1, parameters)  # fully polarised
    g2, dg2 = _pw92_g(rs, 2, parameters)  # parametrises -alpha_c
    fz20 = parameters["fz20"]

    fz = f_zeta(zeta)
    dfz = f_zeta_derivative(zeta)
    z3, z4 = zeta**3, zeta**4

    # eps_c = G0 + z^4 f (G1 - G0 + G2/f''(0)) - f G2/f''(0)
    bracket = g1 - g0 + g2 / fz20
    eps = g0 + z4 * fz * bracket - fz * g2 / fz20

    d_drs = dg0 + z4 * fz * (dg1 - dg0 + dg2 / fz20) - fz * dg2 / fz20
    d_dzeta = (4.0 * z3 * fz + z4 * dfz) * bracket - dfz * g2 / fz20

    return _assemble_vrho(eps, d_drs, d_dzeta, total, rs, zeta, active)


# ------------------------------------------------------------------------- PZ81
def correlation_pz81(rho) -> tuple[np.ndarray, np.ndarray]:
    """Perdew-Zunger 1981 correlation (PRB 23, 5048).

    Two branches meet at ``rs = 1``: a Pade form fitted to the Ceperley-Alder Monte Carlo
    data for the low-density side, and the high-density RPA expansion below it. The
    original coefficients make the value continuous but not the derivative; that kink is
    a property of the functional, not of this implementation.
    """
    rho = _as_spin_density(rho)
    total, rs, zeta = rs_and_zeta(rho)
    active = total > DENSITY_THRESHOLD

    branches = []
    for index in range(2):
        gamma = PZ81_PARAMETERS["gamma"][index]
        b1, b2 = PZ81_PARAMETERS["beta1"][index], PZ81_PARAMETERS["beta2"][index]
        a, b = PZ81_PARAMETERS["a"][index], PZ81_PARAMETERS["b"][index]
        c, d = PZ81_PARAMETERS["c"][index], PZ81_PARAMETERS["d"][index]

        sqrt_rs = np.sqrt(rs)
        denominator = 1.0 + b1 * sqrt_rs + b2 * rs
        low_density = gamma / denominator
        d_low = -gamma * (0.5 * b1 / sqrt_rs + b2) / denominator**2

        log_rs = np.log(np.maximum(rs, 1e-300))
        high_density = a * log_rs + b + c * rs * log_rs + d * rs
        d_high = a / rs + c * (log_rs + 1.0) + d

        dense = rs < 1.0
        branches.append(
            (np.where(dense, high_density, low_density), np.where(dense, d_high, d_low))
        )

    (eps0, deps0), (eps1, deps1) = branches
    fz = f_zeta(zeta)
    dfz = f_zeta_derivative(zeta)

    eps = eps0 + fz * (eps1 - eps0)
    d_drs = deps0 + fz * (deps1 - deps0)
    d_dzeta = dfz * (eps1 - eps0)

    return _assemble_vrho(eps, d_drs, d_dzeta, total, rs, zeta, active)


def _assemble_vrho(eps, d_drs, d_dzeta, total, rs, zeta, active):
    """Chain ``d eps/d rs`` and ``d eps/d zeta`` into ``v_sigma = d(n eps)/d n_sigma``.

    With ``rs ~ n^(-1/3)`` and ``zeta = (n_up - n_dn)/n``::

        d rs/d n     = -rs / (3 n)
        d zeta/d n_up = (1 - zeta)/n,   d zeta/d n_dn = -(1 + zeta)/n

    so ``v_sigma = eps - (rs/3) d eps/d rs + (s - zeta) d eps/d zeta`` with ``s = +1`` for
    spin up and ``-1`` for spin down.
    """
    common = eps - (rs / 3.0) * d_drs
    vrho = np.stack([common + (1.0 - zeta) * d_dzeta, common - (1.0 + zeta) * d_dzeta], axis=-1)
    eps = np.where(active, eps, 0.0)
    vrho = np.where(active[:, None], vrho, 0.0)
    return eps, vrho
