"""Evaluate an exchange-correlation functional on the real-space grid.

This is the layer the SCF loop talks to. It takes a density on the FFT grid, forms the
gradients a GGA needs, calls the pointwise functional, and assembles the local potential

    v_xc^sigma(r) = df/dn_sigma - div( 2 vsigma_ss grad n_sigma + vsigma_ud grad n_sigmabar )

The divergence term is what makes a GGA potential different from an LDA one, and it is
the piece that is easiest to get subtly wrong: it is evaluated in reciprocal space, where
the derivative is exact for a band-limited field rather than a finite-difference stencil.

Spin
----
``density`` has shape ``(nspin, *grid.shape)`` with ``nspin`` 1 or 2. The non-spin-polarised
path is not a separate implementation: it splits ``n/2`` into both channels, runs the same
spin-polarised code, and returns the single (identical) potential. That way there is only
one code path to validate, and the zeta = 0 limit is exact by construction rather than by
a second derivation that could disagree.

Units are Hartree atomic units throughout; the returned energy is in Hartree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..basis.fft import FFTGrid
from . import gga, lda

__all__ = ["XCFunctional", "XCResult", "available_functionals"]

#: Input name -> (kind, exchange/correlation implementation selector).
_FUNCTIONALS: dict[str, tuple[str, str]] = {
    "LDA-PZ": ("lda", "pz81"),
    "LDA-PW": ("lda", "pw92"),
    "PBE": ("gga", "PBE"),
    "PBESOL": ("gga", "PBEsol"),
}


def available_functionals() -> tuple[str, ...]:
    """Names accepted by :class:`XCFunctional`, matching the ``xc.functional`` input."""
    return ("LDA-PZ", "LDA-PW", "PBE", "PBEsol")


@dataclass
class XCResult:
    """Exchange-correlation energy and potential on the grid."""

    #: Total exchange-correlation energy in Hartree.
    energy: float
    #: Local potential in Hartree, shape ``(nspin, *grid.shape)``.
    potential: np.ndarray
    #: Energy per particle on the grid, shape ``grid.shape``; useful for diagnostics.
    energy_per_particle: np.ndarray
    #: GGA only: ``dE/d sigma`` for the three spin pairs, shape ``(3, *grid.shape)``.
    #: ``None`` for an LDA, where the energy does not depend on the gradient.
    vsigma: np.ndarray | None = None
    #: GGA only: ``grad n_s`` per spin channel, shape ``(2, *grid.shape, 3)``.
    #: Kept because the stress tensor needs the outer product ``grad n grad n``, which
    #: cannot be recovered from the potential once the divergence has been taken.
    gradients: np.ndarray | None = None


class XCFunctional:
    """One exchange-correlation functional, evaluated pointwise or on a grid."""

    def __init__(self, name: str):
        key = name.strip().upper().replace("_", "-")
        if key not in _FUNCTIONALS:
            raise ValueError(
                f"unknown functional {name!r}; available: {', '.join(available_functionals())}"
            )
        self.kind, self._selector = _FUNCTIONALS[key]
        self.name = {"PBESOL": "PBEsol"}.get(key, key)

    @property
    def is_gga(self) -> bool:
        return self.kind == "gga"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"XCFunctional({self.name!r})"

    # ------------------------------------------------------------------ pointwise
    def evaluate(self, rho, sigma=None):
        """Exchange plus correlation at a set of points.

        Returns ``(eps, vrho)`` for an LDA and ``(eps, vrho, vsigma)`` for a GGA, with
        libxc's array layouts: ``rho`` is ``(npoints, 2)``, ``sigma`` is ``(npoints, 3)``.
        """
        if self.kind == "lda":
            eps_x, vrho_x = lda.exchange(rho)
            if self._selector == "pz81":
                eps_c, vrho_c = lda.correlation_pz81(rho)
            else:
                eps_c, vrho_c = lda.correlation_pw92(rho)
            return eps_x + eps_c, vrho_x + vrho_c

        if sigma is None:
            raise ValueError(f"{self.name} is a GGA and needs sigma")
        eps_x, vrho_x, vsigma_x = gga.exchange(rho, sigma, self._selector)
        eps_c, vrho_c, vsigma_c = gga.correlation(rho, sigma, self._selector)
        return eps_x + eps_c, vrho_x + vrho_c, vsigma_x + vsigma_c

    # --------------------------------------------------------------------- on grid
    def evaluate_on_grid(self, density: np.ndarray, grid: FFTGrid) -> XCResult:
        """Energy and local potential for a density on the real-space grid.

        ``density`` is ``(nspin, *grid.shape)`` in electrons/Bohr^3.
        """
        density = np.asarray(density, dtype=float)
        if density.ndim == 3:
            density = density[None, ...]
        nspin = density.shape[0]
        if nspin not in (1, 2) or density.shape[1:] != tuple(grid.shape):
            raise ValueError(f"density must have shape (1 or 2, {grid.shape}), got {density.shape}")

        # One spin-polarised code path; the unpolarised case is n/2 in both channels.
        spin_density = np.stack([density[0] / 2, density[0] / 2]) if nspin == 1 else density
        points = int(np.prod(grid.shape))
        rho = spin_density.reshape(2, points).T

        gradients = None
        if self.is_gga:
            gradients = np.stack([grid.gradient(spin_density[s]) for s in range(2)])
            flat = gradients.reshape(2, points, 3)
            sigma = np.stack(
                [
                    np.einsum("pi,pi->p", flat[0], flat[0]),
                    np.einsum("pi,pi->p", flat[0], flat[1]),
                    np.einsum("pi,pi->p", flat[1], flat[1]),
                ],
                axis=-1,
            )
            eps, vrho, vsigma = self.evaluate(rho, sigma)
        else:
            eps, vrho = self.evaluate(rho)
            vsigma = None

        total_density = rho[:, 0] + rho[:, 1]
        energy = float(np.sum(eps * total_density) * grid.cell_volume_per_point)

        potential = vrho.T.reshape(2, *grid.shape)
        if vsigma is not None:
            vsigma_grid = vsigma.T.reshape(3, *grid.shape)
            for s in range(2):
                # df/d(grad n_s) = 2 vsigma_ss grad n_s + vsigma_ud grad n_sbar
                flux = (
                    2.0 * vsigma_grid[2 * s][..., None] * gradients[s]
                    + vsigma_grid[1][..., None] * gradients[1 - s]
                )
                potential[s] = potential[s] - grid.divergence(flux)

        if nspin == 1:
            potential = potential[:1]
        return XCResult(
            energy=energy,
            potential=potential,
            energy_per_particle=eps.reshape(grid.shape),
            vsigma=None if vsigma is None else vsigma.T.reshape(3, *grid.shape),
            gradients=gradients,
        )
