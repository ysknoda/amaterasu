"""The one-centre (on-site) PAW corrections: ``E^a - E~^a`` and its gradient.

This is where PAW earns its accuracy. The plane-wave part of the calculation sees only
smooth functions; everything that happens near the nucleus -- the orthogonality wiggles,
the core-valence exchange, the nuclear attraction -- is restored here, on a radial mesh
inside the augmentation sphere, from the all-electron partial waves the dataset carries.

The energy follows Mortensen, Hansen and Jacobsen, Phys. Rev. B 71, 035109 (2005),
equations (19), (20) and (22), which is the formulation the PAW-XML datasets are written
for. Dropping the frozen-core constants (they do not depend on the occupancies and so
shift every energy by the same amount per atom), the correction is

    E^a - E~^a = sum_ij D_ij dT_ij
               + [ 1/2 <n^a|v[n^a]> - Z int n^a/r ]  -  1/2 <n~^a + Z~^a|v[n~^a + Z~^a]>
               + E_xc[n^a]  -  E_xc[n~^a]
               - int v_bar n~^a ,

with ``n^a = sum_ij D_ij phi_i phi_j + n_c`` and ``n~^a`` the same with the smooth partial
waves and the smooth core. ``Z~^a = sum_L Q_L g_l(r) Y_L`` is the compensation charge.

Angular bookkeeping
-------------------
Electrostatics is done multipole by multipole: every density is expanded in real spherical
harmonics via Gaunt coefficients, and each ``Y_L`` component gets its own exact radial
Poisson solution. Exchange-correlation cannot be expanded that way because it is not
linear in the density, so it is evaluated pointwise on a product grid of radial points
times directions on the sphere.

The exchange-correlation *gradient* with respect to ``D_ij`` is taken in the
integrated-by-parts form

    dE_xc/dD_ij = integral [ df/dn (phi_i phi_j) + df/d(grad n) . grad(phi_i phi_j) ] ,

which needs only the pointwise ``vrho`` and ``vsigma`` the functional already returns. The
alternative -- forming ``v_xc = df/dn - div(df/d grad n)`` on the sphere -- would require a
divergence in spherical coordinates, which is both more code and less accurate.

Every derivative here is checked against a finite difference in ``D_ij`` in the test suite.
That is the decisive test for this module: the Hamiltonian is *defined* as the gradient of
the energy, so if the two agree, the two most intricate parts of PAW are consistent, and
the self-consistent solution they define is the stationary point of the energy that is
actually being reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ...xc.driver import XCFunctional
from .angular import SphericalGrid, gaunt_coefficients, lm_index
from .dataset import PAWDataset
from .radial import derivative, integrate, poisson_operator
from .shape import GaussianShape

__all__ = ["OnsiteResult", "OnsiteSetup"]

#: Angular resolution for the on-site exchange-correlation integration.
#:
#: The electrostatic terms are expanded in spherical harmonics and are exact; only
#: exchange-correlation is integrated pointwise, and its integrand is not a polynomial, so
#: this is a resolution choice rather than an exactness guarantee. A *spherical* on-site
#: density -- the isolated-atom limit, and close to what a real solid has -- is integrated
#: exactly at any degree, because the integrand is then constant over the sphere. On a
#: deliberately extreme aspherical test density the measured error against a degree-40
#: rule is 2.0 meV at degree 8, 0.26 meV at 12, 0.14 meV at 16 and 0.04 meV at 20, for a
#: cost of 45, 91, 153 and 231 directions. Degree 20 keeps the worst case an order of
#: magnitude inside the millielectronvolt budget; ``OnsiteSetup(angular_degree=...)``
#: trades it back for speed.
DEFAULT_ANGULAR_DEGREE = 20


@dataclass
class OnsiteResult:
    """One atom's on-site correction and its gradient."""

    #: ``E^a - E~^a`` in Hartree, without the frozen-core constants.
    energy: float
    #: ``d(E^a - E~^a)/dD_ij``, shape ``(nspin, ni, ni)`` in Hartree.
    hamiltonian: np.ndarray
    #: Compensation-charge multipole moments ``Q_L``, shape ``((lmax+1)^2,)``.
    multipoles: np.ndarray
    #: Energy of each term, for diagnostics; the values sum to :attr:`energy`.
    decomposition: dict = field(default_factory=dict)
    #: The gradient of each term, which the tests check against a finite difference of
    #: the matching entry in :attr:`decomposition` -- term by term, so a discrepancy
    #: points at one expression rather than at the sum.
    gradient_terms: dict = field(default_factory=dict)


@dataclass
class OnsiteSetup:
    """Everything about one species' augmentation sphere that the SCF loop reuses.

    Built once per species. The expensive parts -- Gaunt coefficients, the partial waves
    evaluated on the angular grid, the multipole coefficients -- depend only on the dataset
    and are cached here; only the occupancy matrix changes from iteration to iteration.
    """

    dataset: PAWDataset
    xc: XCFunctional
    shape: GaussianShape
    angular_degree: int = DEFAULT_ANGULAR_DEGREE

    def __post_init__(self):
        data = self.dataset
        # Far enough out that the compensation charge is entirely enclosed, not merely to
        # r_c. The partial waves and the two core densities have already cancelled by r_c,
        # so the extra shell contributes only the shape-function tail -- which the smooth
        # energy carries over all space and which therefore has to be subtracted here too.
        self.stop = data.points_within(max(data.rcut, self.shape.containment_radius()))
        self.r = data.grid.r[: self.stop]
        self.lmax_wave = data.l_max
        self.lmax_density = 2 * self.lmax_wave

        # Expand each radial channel over its 2l+1 magnetic sublevels.
        self.channel_of: list[int] = []
        self.lm_of: list[int] = []
        for index, wave in enumerate(data.waves):
            for m in range(-wave.l, wave.l + 1):
                self.channel_of.append(index)
                self.lm_of.append(lm_index(wave.l, m))
        self.channel_of = np.array(self.channel_of, dtype=int)
        self.lm_of = np.array(self.lm_of, dtype=int)
        self.n_projectors = len(self.lm_of)

        ae = data.stacked("ae")[:, : self.stop]
        ps = data.stacked("ps")[:, : self.stop]
        self.radial_ae = ae[self.channel_of]
        self.radial_ps = ps[self.channel_of]

        # dT is diagonal in (l, m) and carries the radial channel coupling.
        same_lm = self.lm_of[:, None] == self.lm_of[None, :]
        self.kinetic = data.kinetic_differences[np.ix_(self.channel_of, self.channel_of)] * same_lm

        gaunt = gaunt_coefficients(self.lmax_wave, self.lmax_density)
        self.gaunt = gaunt[:, self.lm_of[:, None], self.lm_of[None, :]]

        # Delta_{L,ij} = int Y_L r^l [phi_i phi_j - phit_i phit_j], Eq. (15).
        pair = self.radial_ae[:, None, :] * self.radial_ae[None, :, :]
        pair = pair - self.radial_ps[:, None, :] * self.radial_ps[None, :, :]
        self.delta = np.zeros((self.n_lm, self.n_projectors, self.n_projectors))
        for l in range(self.lmax_density + 1):
            moment = integrate(data.grid, pair * self.r ** (l + 2), upto=self.stop)
            for m in range(-l, l + 1):
                index = lm_index(l, m)
                self.delta[index] = moment * self.gaunt[index]

        # Delta^a, Eq. (16): the nuclear charge plus the core density difference.
        core_difference = integrate(
            data.grid,
            (data.ae_core_density - data.pseudo_core_density)[: self.stop] * self.r**2,
            upto=self.stop,
        )
        self.delta0 = float(-data.atomic_number / np.sqrt(4.0 * np.pi) + core_difference)

        # O_ij = sqrt(4 pi) Delta_{00,ij}, Eq. (39): the PAW overlap operator's coefficients.
        self.overlap = np.sqrt(4.0 * np.pi) * self.delta[0]

        self.core_ae = data.ae_core_density[: self.stop]
        self.core_ps = data.pseudo_core_density[: self.stop]
        self.vbar = data.zero_potential[: self.stop]

    # ------------------------------------------------------------------ cached tables
    @property
    def n_lm(self) -> int:
        return (self.lmax_density + 1) ** 2

    @cached_property
    def angular(self) -> SphericalGrid:
        return SphericalGrid.build(self.angular_degree)

    @cached_property
    def _harmonics(self) -> np.ndarray:
        """``Y_{L_i}`` at each direction, shape ``(nproj, nangular)``."""
        return self.angular.harmonics(self.lmax_wave)[self.lm_of]

    @cached_property
    def _harmonic_gradients(self) -> np.ndarray:
        """``grad_Omega Y_{L_i}``, shape ``(nproj, nangular, 3)``."""
        return self.angular.gradients(self.lmax_wave)[self.lm_of]

    @cached_property
    def _shape_radial(self) -> np.ndarray:
        """``g_l(r)`` for every ``l`` up to the density expansion, shape ``(lmax+1, nr)``."""
        return np.array([self.shape.radial(l, self.r) for l in range(self.lmax_density + 1)])

    def _basis(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        """``u_i(r, Omega)`` and ``grad u_i``, shapes ``(nproj, nr, nang)`` and ``(..., 3)``."""
        radial = self.radial_ae if which == "ae" else self.radial_ps
        slope = derivative(self.dataset.grid, radial[:, : self.stop])
        harmonics, gradients = self._harmonics, self._harmonic_gradients
        inverse_r = np.where(self.r > 0, 1.0 / np.where(self.r > 0, self.r, 1.0), 0.0)

        values = radial[:, :, None] * harmonics[:, None, :]
        directions = self.angular.directions
        grad = slope[:, :, None, None] * harmonics[:, None, :, None] * directions[None, None, :, :]
        grad += (radial * inverse_r)[:, :, None, None] * gradients[:, None, :, :]
        return values, grad

    @cached_property
    def _basis_ae(self):
        return self._basis("ae")

    @cached_property
    def _basis_ps(self):
        return self._basis("ps")

    def __str__(self) -> str:
        return (
            f"{self.dataset.symbol} augmentation sphere: {self.n_projectors} projectors, "
            f"density to l={self.lmax_density}, {self.stop} radial x {len(self.angular)} "
            f"angular points, {self.shape}"
        )

    def atomic_occupancies(self, nspin: int = 1, moment: float = 0.0) -> np.ndarray:
        """The generating atom's filling, spread evenly over each shell's ``m`` values.

        This is the natural partner of the superposed atomic density that seeds the SCF:
        both describe the same non-interacting collection of atoms, so the compensation
        charge of the first iteration is consistent with the density it compensates.
        """
        matrix = np.zeros((nspin, self.n_projectors, self.n_projectors))
        total = sum(w.occupation for w in self.dataset.waves)
        for row, channel in enumerate(self.channel_of):
            wave = self.dataset.waves[channel]
            if not wave.occupation:
                continue
            filling = wave.occupation / (2 * wave.l + 1)
            if nspin == 1:
                matrix[0, row, row] = filling
            else:
                share = 0.0 if total <= 0 else moment * wave.occupation / total
                for spin, sign in ((0, +1.0), (1, -1.0)):
                    matrix[spin, row, row] = 0.5 * (filling + sign * share / (2 * wave.l + 1))
        return matrix

    # ------------------------------------------------------------------ quadrature
    @cached_property
    def poisson(self) -> list[np.ndarray]:
        """One symmetrised radial Poisson operator per ``l``, built once per species."""
        measure = self.radial_weights * self.r**2
        return [
            poisson_operator(self.dataset.grid, l, upto=self.stop, measure=measure)
            for l in range(self.lmax_density + 1)
        ]

    @cached_property
    def radial_weights(self) -> np.ndarray:
        """Quadrature weights ``w_r`` with ``sum_r w_r f_r == integral f dr``.

        Obtained by applying :func:`~amaterasu.pseudo.paw.radial.integrate` to the identity
        matrix, so the weights cannot drift out of step with the integrator itself.
        """
        return integrate(self.dataset.grid, np.eye(self.stop), upto=self.stop)

    @cached_property
    def _volume_weights(self) -> np.ndarray:
        """``w_r r^2 w_Omega``, shape ``(nr, nangular)``: the measure of ``d^3r``."""
        return (self.radial_weights * self.r**2)[:, None] * self.angular.weights[None, :]

    # ------------------------------------------------------------- density expansion
    def multipoles(self, occupancies: np.ndarray) -> np.ndarray:
        """``Q_L``, the compensation charge's electrostatic moments (Eq. 14)."""
        total = np.asarray(occupancies, dtype=float).sum(axis=0)
        moments = np.einsum("Lij,ij->L", self.delta, total)
        moments[0] += self.delta0
        return moments

    def _components(self, occupancies: np.ndarray, which: str) -> np.ndarray:
        """``Y_L`` components of the one-centre density, shape ``(n_lm, nr)``."""
        radial = self.radial_ae if which == "ae" else self.radial_ps
        core = self.core_ae if which == "ae" else self.core_ps
        total = np.asarray(occupancies, dtype=float).sum(axis=0)
        pair = radial[:, None, :] * radial[None, :, :]
        components = np.einsum("Lij,ij,ijr->Lr", self.gaunt, total, pair)
        components[0] = components[0] + core
        return components

    # --------------------------------------------------------------- the correction
    def compute(self, occupancies: np.ndarray) -> OnsiteResult:
        """``E^a - E~^a`` and ``d/dD_ij`` for one atom's occupancy matrix.

        ``occupancies`` has shape ``(nspin, nproj, nproj)``. As everywhere else in
        amaterasu, an unpolarised calculation carries the *total* occupancy in its single
        channel, so ``nspin = 1`` and ``nspin = 2`` describing the same state give the same
        energy -- a symmetry the test suite checks.
        """
        occupancies = np.atleast_3d(np.asarray(occupancies, dtype=float))
        if occupancies.shape[1:] != (self.n_projectors, self.n_projectors):
            raise ValueError(
                f"occupancies have shape {occupancies.shape}, expected "
                f"(nspin, {self.n_projectors}, {self.n_projectors})"
            )
        occupancies = 0.5 * (occupancies + occupancies.transpose(0, 2, 1))
        r = self.r
        weights = self.radial_weights

        moments = self.multipoles(occupancies)
        ae_components = self._components(occupancies, "ae")
        ps_components = self._components(occupancies, "ps")

        # The compensation charge, one multipole at a time.
        compensated = ps_components.copy()
        for index in range(self.n_lm):
            l = int(np.sqrt(index))
            compensated[index] = compensated[index] + moments[index] * self._shape_radial[l]

        # ---------------------------------------------------------------- electrostatics
        measure = weights * r**2
        ae_potential = np.empty_like(ae_components)
        ps_potential = np.empty_like(compensated)
        for index in range(self.n_lm):
            operator = self.poisson[int(np.sqrt(index))]
            ae_potential[index] = operator @ ae_components[index]
            ps_potential[index] = operator @ compensated[index]
        energy_ae = 0.5 * float(np.sum(ae_components * ae_potential * measure))
        energy_ps = 0.5 * float(np.sum(compensated * ps_potential * measure))

        # Electron-nucleus attraction: v_nuc = -Z/r, whose only Y_L component is
        # -Z sqrt(4 pi)/r, so the r^2 of the volume element leaves a harmless r.
        nuclear_weight = -self.dataset.atomic_number * np.sqrt(4.0 * np.pi) * weights * r
        energy_nuclear = float(np.sum(ae_components[0] * nuclear_weight))

        # ------------------------------------------------------------------ zero potential
        energy_vbar = float(np.sum(ps_components[0] * self.vbar * measure))

        # ------------------------------------------------------------------------- xc
        xc_ae_energy, xc_ae_hamiltonian = self._exchange_correlation(occupancies, "ae")
        xc_ps_energy, xc_ps_hamiltonian = self._exchange_correlation(occupancies, "ps")

        energy = (
            float(np.einsum("sij,ij->", occupancies, self.kinetic))
            + (energy_ae + energy_nuclear - energy_ps)
            + (xc_ae_energy - xc_ps_energy)
            - energy_vbar
        )

        # --------------------------------------------------------------- the gradient
        pair_ae = self.radial_ae[:, None, :] * self.radial_ae[None, :, :]
        pair_ps = self.radial_ps[:, None, :] * self.radial_ps[None, :, :]

        # int v_L phi_i phi_j r^2 dr, contracted with the Gaunt coefficients.
        ae_matrix = np.einsum("Lij,Lr,ijr->ij", self.gaunt, ae_potential * measure, pair_ae)
        ae_matrix = ae_matrix + self.gaunt[0] * np.einsum("r,ijr->ij", nuclear_weight, pair_ae)
        ps_matrix = np.einsum("Lij,Lr,ijr->ij", self.gaunt, ps_potential * measure, pair_ps)
        # The compensation charge also moves with D, through Q_L.
        ps_matrix = ps_matrix + np.einsum(
            "Lij,Lr,Lr->ij", self.delta, ps_potential * measure, self._shape_expanded
        )
        vbar_matrix = self.gaunt[0] * np.einsum("r,ijr->ij", self.vbar * measure, pair_ps)

        electrostatic = ae_matrix - ps_matrix - vbar_matrix
        hamiltonian = (
            self.kinetic[None, :, :]
            + electrostatic[None, :, :]
            + xc_ae_hamiltonian
            - xc_ps_hamiltonian
        )

        return OnsiteResult(
            energy=energy,
            hamiltonian=np.ascontiguousarray(hamiltonian),
            multipoles=moments,
            decomposition={
                "kinetic": float(np.einsum("sij,ij->", occupancies, self.kinetic)),
                "electrostatic": energy_ae + energy_nuclear - energy_ps,
                "xc": xc_ae_energy - xc_ps_energy,
                "vbar": -energy_vbar,
            },
            gradient_terms={
                "kinetic": self.kinetic,
                "electrostatic": ae_matrix - ps_matrix,
                "xc": xc_ae_hamiltonian - xc_ps_hamiltonian,
                "vbar": -vbar_matrix,
            },
        )

    @cached_property
    def _shape_expanded(self) -> np.ndarray:
        """``g_l(r)`` indexed by the composite ``L`` rather than by ``l``."""
        return np.array([self._shape_radial[int(np.sqrt(index))] for index in range(self.n_lm)])

    def _exchange_correlation(self, occupancies, which: str) -> tuple[float, np.ndarray]:
        """``E_xc`` and ``dE_xc/dD_ij`` for the all-electron or the smooth density."""
        values, gradients = self._basis_ae if which == "ae" else self._basis_ps
        core = self.core_ae if which == "ae" else self.core_ps
        nspin = occupancies.shape[0]
        y00 = 1.0 / np.sqrt(4.0 * np.pi)
        shape = values.shape[1:]

        # Flattened so the expensive contraction is a matrix product rather than a
        # four-index einsum: with a d-shell dataset on the default angular grid the naive
        # form spends more time here than in the exchange-correlation functional itself.
        flat_values = values.reshape(self.n_projectors, -1)
        flat_gradients = gradients.reshape(self.n_projectors, -1, 3)
        coupled = np.einsum("sij,jp->sip", occupancies, flat_values, optimize=True)

        density = np.einsum("sip,ip->sp", coupled, flat_values, optimize=True).reshape(
            (nspin, *shape)
        )
        density = density + (core * y00)[None, :, None] / nspin

        core_slope = derivative(self.dataset.grid, core) * y00
        gradient = 2.0 * np.einsum("sip,ipc->spc", coupled, flat_gradients, optimize=True).reshape(
            (nspin, *shape, 3)
        )
        gradient = gradient + (core_slope[:, None, None] / nspin) * self.angular.directions

        spin_density = np.concatenate([density, density]) / 2.0 if nspin == 1 else density
        spin_gradient = np.concatenate([gradient, gradient]) / 2.0 if nspin == 1 else gradient

        points = spin_density.shape[1] * spin_density.shape[2]
        rho = spin_density.reshape(2, points).T
        flat = spin_gradient.reshape(2, points, 3)
        if self.xc.is_gga:
            sigma = np.stack(
                [
                    np.einsum("pi,pi->p", flat[0], flat[0]),
                    np.einsum("pi,pi->p", flat[0], flat[1]),
                    np.einsum("pi,pi->p", flat[1], flat[1]),
                ],
                axis=-1,
            )
            eps, vrho, vsigma = self.xc.evaluate(rho, sigma)
        else:
            eps, vrho = self.xc.evaluate(rho)
            vsigma = None

        shape = spin_density.shape[1:]
        measure = self._volume_weights
        energy = float(np.sum(eps.reshape(shape) * spin_density.sum(axis=0) * measure))

        # Effective potentials conjugate to a change in D. For nspin == 1 both channels
        # move together and each takes half the change, which averages the two responses.
        vrho_grid = vrho.T.reshape(2, *shape)
        if vsigma is not None:
            vsigma_grid = vsigma.T.reshape(3, *shape)
            flux = np.stack(
                [
                    2.0 * vsigma_grid[2 * s][..., None] * spin_gradient[s]
                    + vsigma_grid[1][..., None] * spin_gradient[1 - s]
                    for s in range(2)
                ]
            )
        else:
            flux = np.zeros((2, *shape, 3))
        if nspin == 1:
            vrho_grid = 0.5 * (vrho_grid[0] + vrho_grid[1])[None, ...]
            flux = 0.5 * (flux[0] + flux[1])[None, ...]

        # Contracted without materialising the (i, j, r, k) pair density, which for a
        # d-shell dataset on the default angular grid would be tens of megabytes.
        flat_measure = measure.ravel()
        scaled = flat_values * flat_measure
        weighted = vrho_grid.reshape(nspin, -1)[:, None, :] * scaled[None, :, :]
        hamiltonian = weighted @ flat_values.T
        if vsigma is not None:
            flux_flat = flux.reshape(nspin, -1, 3) * flat_measure[None, :, None]
            projected = np.einsum("spc,ipc->sip", flux_flat, flat_gradients, optimize=True)
            cross = projected @ flat_values.T
            hamiltonian = hamiltonian + cross + cross.transpose(0, 2, 1)
        return energy, hamiltonian
