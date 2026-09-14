"""The projector-augmented-wave dataset: partial waves, projectors and frozen core.

A PAW dataset is a much richer object than a norm-conserving pseudopotential. Alongside
the smooth quantities a plane-wave code needs, it carries the *all-electron* partial waves
inside the augmentation sphere, which is what lets PAW reconstruct the true wave function
and reach all-electron accuracy for valence properties.

Radial conventions, verified numerically against a real file rather than assumed
-------------------------------------------------------------------------------
Every radial function is stored as it appears in the PAW-XML file, which is the ``Y_L``
expansion coefficient rather than the function itself:

``ae``, ``ps``, ``projector``
    The plain radial parts ``phi(r)``, ``phi~(r)``, ``p~(r)`` -- *not* ``r`` times them.
    The full functions are these multiplied by ``Y_lm``, and the biorthogonality is
    ``integral p~_i phi~_j r^2 dr = delta_ij``, checked by :meth:`biorthogonality_error`.
``ae_core_density``, ``pseudo_core_density``, ``pseudo_valence_density``, ``zero_potential``
    The ``l = 0`` coefficients: the real quantity is these times ``Y_00 = 1/sqrt(4 pi)``,
    so a core density integrates as ``sqrt(4 pi) integral n(r) r^2 dr = N_core``.

Keeping the file's own convention throughout the one-centre code means every radial array
is an ``n_L(r)``, which is exactly what the radial Poisson solver and the Gaunt
bookkeeping expect. The conversion to the ``4 pi r^2 rho`` convention that the plane-wave
side uses happens once, in :meth:`as_plane_wave_density`.

Partial waves outside the sphere
--------------------------------
By construction ``phi_i(r) = phi~_i(r)`` for ``r > r_c`` -- bitwise, in the files checked
here -- so every one-centre *difference* vanishes there, and the region beyond the sphere
contributes nothing to ``E^a - E~^a`` except through the compensation charge. The
one-centre integrals therefore run out to the shape's own ``containment_radius`` rather
than stopping at :attr:`sphere_points`, using :meth:`points_within`. The default Gaussian
has no compact support, so the part of it beyond ``r_c`` is added by the smooth energy and
has to be subtracted here; a compactly supported shape asks for nothing extra, its
containment radius being ``r_c`` itself.

They must not run *much* further, though. Several partial waves are unbound scattering
states (silicon's second ``s`` channel sits at ``+1.5`` Hartree) and diverge at large
``r``. At the Gaussian's containment radius they are still small -- for silicon, under 0.63
against a peak of 98.7 inside the sphere -- but the margin is finite, and anything that
pushes the integration radius out much beyond twice ``r_c`` should be treated with
suspicion. A compactly supported shape does not ask for the margin at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..base import RadialGrid
from .radial import integrate

__all__ = ["PAWDataset", "PartialWave"]


@dataclass(frozen=True)
class PartialWave:
    """One channel: an all-electron partial wave, its smooth counterpart and a projector."""

    l: int
    label: str
    #: Principal quantum number, or ``None`` for an unbound scattering channel.
    n: int | None
    #: Reference occupation of the generating atom; zero for an unbound channel.
    occupation: float
    #: Reference energy in Hartree (negative for a bound state).
    energy: float
    ae: np.ndarray
    ps: np.ndarray
    projector: np.ndarray

    @property
    def is_bound(self) -> bool:
        return self.n is not None

    def __str__(self) -> str:
        shell = f"{self.n}{'spdfgh'[self.l]}" if self.is_bound else f"eps-{'spdfgh'[self.l]}"
        return f"{shell} (f={self.occupation:g}, e={self.energy:+.4f} Ha)"


@dataclass
class PAWDataset:
    """A PAW dataset for one chemical species."""

    symbol: str
    atomic_number: float
    #: Number of frozen core electrons.
    core_electrons: float
    #: Number of valence electrons the dataset treats explicitly.
    valence_charge: float
    grid: RadialGrid
    waves: list[PartialWave]
    #: Augmentation radius in Bohr.
    rcut: float
    #: ``<phi_i|T|phi_j> - <phi~_i|T|phi~_j>`` in Hartree, shape ``(nwaves, nwaves)``.
    kinetic_differences: np.ndarray
    #: ``l = 0`` coefficients (multiply by ``Y_00`` for the real quantity).
    ae_core_density: np.ndarray
    pseudo_core_density: np.ndarray
    pseudo_valence_density: np.ndarray
    zero_potential: np.ndarray
    xc_functional: str | None = None
    xc_description: str = ""
    generator: str = ""
    source: str = ""
    #: Recommended plane-wave cutoffs in Hartree, keyed ``low``/``medium``/``high``.
    ecut_hints: dict[str, float] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    #: Marks this as a PAW dataset wherever the code accepts either kind.
    kind: str = "PAW"

    def __post_init__(self):
        size = len(self.grid)
        for name in (
            "ae_core_density",
            "pseudo_core_density",
            "pseudo_valence_density",
            "zero_potential",
        ):
            values = getattr(self, name)
            if len(values) != size:
                raise ValueError(f"{name} has {len(values)} points but the grid has {size}")
        expected = (len(self.waves), len(self.waves))
        if self.kinetic_differences.shape != expected:
            raise ValueError(
                f"kinetic_energy_differences has shape {self.kinetic_differences.shape}, "
                f"expected {expected} for {len(self.waves)} partial waves"
            )

    def __str__(self) -> str:
        channels = ", ".join(str(w) for w in self.waves)
        return (
            f"{self.symbol} PAW (Z={self.atomic_number:g}, {self.core_electrons:g} core + "
            f"{self.valence_charge:g} valence) {self.xc_description or 'unknown XC'}, "
            f"r_c={self.rcut:.3f} Bohr, {len(self.waves)} partial waves [{channels}]"
        )

    # ------------------------------------------------------------------ geometry
    @property
    def l_max(self) -> int:
        return max(w.l for w in self.waves) if self.waves else 0

    @property
    def n_waves(self) -> int:
        return len(self.waves)

    @property
    def sphere_points(self) -> int:
        """Number of mesh points inside the augmentation sphere, one past ``r_c``.

        This is where the partial waves stop meaning anything: outside it the all-electron
        and smooth ones coincide by construction. One extra point is included so that the
        quadrature panel reaching ``r_c`` is complete.
        """
        return self.points_within(self.rcut)

    def points_within(self, radius: float) -> int:
        """Number of mesh points out to ``radius``, one past it.

        The one-centre integrals run further than ``sphere_points`` when the compensation
        charge does; see the shape's ``containment_radius``, which for the default Gaussian
        is beyond ``r_c`` and for a compactly supported shape is ``r_c`` itself.
        """
        inside = int(np.searchsorted(self.grid.r, float(radius), side="right"))
        return min(inside + 1, len(self.grid))

    @property
    def wave_l(self) -> np.ndarray:
        return np.array([w.l for w in self.waves], dtype=int)

    def stacked(self, attribute: str) -> np.ndarray:
        """``(nwaves, npoints)`` array of one radial function from every partial wave."""
        return np.array([getattr(w, attribute) for w in self.waves])

    # --------------------------------------------------------------- diagnostics
    def biorthogonality_error(self) -> float:
        """``max |<p~_i|phi~_j> - delta_ij|`` within each ``l`` channel.

        The single cheapest check that the file was read correctly: it fails immediately
        if the radial convention, the mesh or the Jacobian is wrong.
        """
        stop = self.sphere_points
        projectors = self.stacked("projector")[:, :stop]
        smooth = self.stacked("ps")[:, :stop]
        weight = self.grid.r[:stop] ** 2
        overlap = integrate(
            self.grid, projectors[:, None, :] * smooth[None, :, :] * weight, upto=stop
        )
        same_l = self.wave_l[:, None] == self.wave_l[None, :]
        return float(np.max(np.abs(overlap - np.eye(self.n_waves))[same_l]))

    def core_charge_error(self) -> float:
        """``sqrt(4 pi) integral n_c r^2 dr`` against the declared number of core electrons."""
        charge = np.sqrt(4.0 * np.pi) * self.grid.integrate(self.ae_core_density * self.grid.r**2)
        return float(abs(charge - self.core_electrons))

    def partial_waves_match_outside(self) -> float:
        """``max |phi_i - phi~_i|`` beyond ``r_c``, relative to the wave's size there.

        Zero by construction; a non-zero value means the augmentation radius in the file
        disagrees with where the partial waves were actually matched.
        """
        stop = self.sphere_points
        if stop >= len(self.grid):  # pragma: no cover - defensive
            return 0.0
        difference = self.stacked("ae")[:, stop:] - self.stacked("ps")[:, stop:]
        scale = max(float(np.max(np.abs(self.stacked("ps")[:, stop:]))), 1e-30)
        return float(np.max(np.abs(difference)) / scale)

    def as_plane_wave_density(self, which: str) -> np.ndarray:
        """A stored ``l = 0`` density in the ``4 pi r^2 rho`` convention used on the grid."""
        values = {
            "core": self.pseudo_core_density,
            "ae_core": self.ae_core_density,
            "valence": self.pseudo_valence_density,
        }[which]
        return np.sqrt(4.0 * np.pi) * values * self.grid.r**2

    # ------------------------------------------------- plane-wave side compatibility
    # The superposition helpers in amaterasu.hamiltonian.local take any object with these
    # three attributes, so a PAW dataset can seed the density and place its smooth core on
    # the grid through exactly the same code path a norm-conserving one does.
    @property
    def core_density(self) -> np.ndarray:
        """The *smooth* core density as ``4 pi r^2 rho``, for the grid.

        In PAW this is not only a nonlinear core correction. The smooth core is part of
        the charge density ``n~`` that the compensation charge is built against, so it
        enters the Hartree term as well as exchange-correlation.
        """
        return self.as_plane_wave_density("core")

    @property
    def valence_density(self) -> np.ndarray:
        """The smooth atomic valence density as ``4 pi r^2 rho``: the SCF starting guess."""
        return self.as_plane_wave_density("valence")

    @property
    def has_core_correction(self) -> bool:
        return True

    def check(self) -> list[str]:
        """Problems found in the dataset; an empty list means it looks sound."""
        problems = []
        error = self.biorthogonality_error()
        if error > 1e-6:
            problems.append(
                f"<p~_i|phi~_j> departs from the identity by {error:.2e}; the radial "
                "convention or the mesh Jacobian is being misread"
            )
        charge = self.core_charge_error()
        if charge > 1e-3:
            problems.append(
                f"the core density integrates {charge:.4f} electrons away from the "
                f"declared {self.core_electrons:g}"
            )
        outside = self.partial_waves_match_outside()
        if outside > 1e-6:
            problems.append(
                f"phi and phi~ differ by {outside:.2e} outside r_c = {self.rcut:.3f} Bohr, "
                "so the one-centre truncation would lose a real contribution"
            )
        if not np.allclose(self.kinetic_differences, self.kinetic_differences.T, atol=1e-8):
            problems.append("the kinetic-energy-difference matrix is not symmetric")
        return problems
