"""Reference exchange-correlation values from libxc, for tests only.

amaterasu implements LDA, PBE and PBEsol itself; this module exists so those
implementations can be checked point-by-point against an independent, widely used one.
Nothing in the production code path may import it.

Installing libxc
----------------
libxc's Python bindings are not on PyPI -- the ``pylibxc`` and ``pylibxc2`` names there are
unrelated stubs. Build from source instead::

    curl -L -o libxc.tar.gz https://gitlab.com/libxc/libxc/-/archive/7.1.2/libxc-7.1.2.tar.gz
    tar xzf libxc.tar.gz && cd libxc-7.1.2
    pip install .          # pulls cmake into an isolated build env; no system cmake needed

Every function here degrades to :func:`available` returning False when libxc is absent, so
the test suite skips rather than fails on a machine without it.

Two findings worth carrying into amaterasu's own implementation
--------------------------------------------------------------
1. libxc's ``gga_c_pbe`` reduces at zero gradient to ``lda_c_pw_mod``, **not** ``lda_c_pw``
   -- exactly, at every density tested. ``lda_c_pw_mod`` is the PW92 parametrisation with
   the extra decimal places, which is what the PBE paper uses. Using plain ``lda_c_pw``
   instead leaves a systematic offset of ~2.5e-7 Ha/electron.
2. Consequently the gradient correction must be extracted as
   ``H = eps_c(sigma) - eps_c(sigma=0)`` **of the same functional**. Subtracting
   ``lda_c_pw`` instead leaves that offset behind, and once ``H`` falls below it the
   extracted beta diverges rather than converging.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PBESOL_PARAMETERS",
    "PBE_PARAMETERS",
    "available",
    "evaluate",
    "exchange_enhancement",
    "extract_correlation_beta",
    "extract_exchange_mu",
    "reduced_gradient_sigma",
    "version",
]

#: PBE parameters (Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)), confirmed against libxc.
PBE_PARAMETERS = {"mu": 0.2195149727645171, "beta": 0.06672455060314922, "kappa": 0.804}

#: PBEsol parameters (Perdew et al., PRL 100, 136406 (2008)), confirmed against libxc.
#: mu is restored to the second-order gradient expansion value 10/81.
PBESOL_PARAMETERS = {"mu": 10.0 / 81.0, "beta": 0.046, "kappa": 0.804}


def available() -> bool:
    """True if libxc's Python bindings can be imported."""
    try:
        import pylibxc  # noqa: F401
    except ImportError:
        return False
    return True


def version() -> str | None:
    """The installed libxc version, or None if it is not available."""
    if not available():
        return None
    import pylibxc

    return pylibxc.version.__version__


def evaluate(name: str, rho, sigma=None, *, spin: str = "unpolarized", key: str = "zk"):
    """Evaluate a libxc functional and return one output array.

    ``key`` selects the quantity: ``zk`` is the energy per particle, ``vrho`` its density
    derivative, ``vsigma`` its derivative with respect to ``|grad n|^2``.
    """
    import pylibxc

    inputs = {"rho": np.atleast_1d(np.asarray(rho, dtype=float))}
    if sigma is not None:
        inputs["sigma"] = np.atleast_1d(np.asarray(sigma, dtype=float))
    result = pylibxc.LibXCFunctional(name, spin).compute(inputs)
    return np.asarray(result[key]).ravel()


def reduced_gradient_sigma(rho: float, s: float) -> float:
    """``sigma = |grad n|^2`` corresponding to the exchange reduced gradient ``s``.

    ``s = |grad n| / (2 k_F n)`` with ``k_F = (3 pi^2 n)^(1/3)``.
    """
    k_f = (3.0 * np.pi**2 * rho) ** (1 / 3)
    return (s * 2.0 * k_f * rho) ** 2


def exchange_enhancement(name: str, rho: float, s: float) -> float:
    """The exchange enhancement factor ``F_x(s) = eps_x^GGA / eps_x^LDA`` at density ``rho``."""
    eps_lda = -0.75 * (3.0 / np.pi) ** (1 / 3) * rho ** (1 / 3)
    return float(evaluate(name, rho, reduced_gradient_sigma(rho, s))[0]) / eps_lda


def extract_exchange_mu(name: str, *, rho: float = 1.0, s: float = 1e-4) -> float:
    """Recover the gradient-expansion coefficient from ``F_x = 1 + mu s^2 + O(s^4)``.

    ``s`` must be small enough that the ``s^4`` term is negligible but large enough that
    ``F_x - 1`` stays well above roundoff; 1e-4 sits comfortably in that window.
    """
    return (exchange_enhancement(name, rho, s) - 1.0) / s**2


def extract_correlation_beta(name: str, *, rho: float = 1.0, s: float = 3e-5) -> float:
    """Recover ``beta`` from ``H -> beta t^2`` as the gradient vanishes.

    ``H`` is taken as the functional's own zero-gradient value subtracted from its value at
    finite gradient. Subtracting ``lda_c_pw`` instead would leave the ~2.5e-7 offset
    described in the module docstring and the estimate would diverge, not converge.
    """
    k_f = (3.0 * np.pi**2 * rho) ** (1 / 3)
    k_s = np.sqrt(4.0 * k_f / np.pi)
    t = s * k_f / k_s  # phi = 1 for an unpolarised density
    h = float(evaluate(name, rho, reduced_gradient_sigma(rho, s))[0]) - float(
        evaluate(name, rho, 0.0)[0]
    )
    return h / t**2
