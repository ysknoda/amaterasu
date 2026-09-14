"""The stress tensor: the derivative of the total energy with respect to a strain.

Convention
----------
``sigma_ab = (1/Omega) dE/d eps_ab`` where the strain maps ``r -> (1 + eps) r``. This is
ASE's convention, so a hydrostatic pressure ``P = -tr(sigma)/3`` is positive when the cell
would like to expand. Units here are Ha/Bohr^3; the calculator converts once at the
boundary.

How each term picks up a strain
-------------------------------
Straining the cell moves three things at once, and every term below is one combination of
them:

* the volume, ``Omega -> Omega (1 + tr eps)``;
* the reciprocal vectors, ``G -> (1 - eps) G``, hence ``d|G|^2/d eps_uv = -2 G_u G_v``;
* the density, which dilutes as ``n -> n/det`` -- *except* the frozen partial core, which
  is rigidly attached to its atom and only translates. That exception is a separate term.

Wavefunction coefficients are held fixed. That is exact at self-consistency by the
variational principle: the energy is stationary with respect to them, so their strain
response contributes nothing to first order.

Pulay stress
------------
Unlike the forces, the stress is *not* free of a basis-set error. The plane-wave basis is
defined by ``|k+G|^2/2 <= ecut`` in the strained cell, so straining changes which functions
exist. The derivative above holds the set of Miller indices fixed and therefore misses
that discontinuous change; what is missing is the Pulay stress, a spurious pressure that
vanishes only as the cutoff converges. :func:`pulay_stress_estimate` measures it directly
by repeating the calculation at a higher cutoff, and the variable-cell relaxation driver
warns about it rather than leaving it implicit.
"""

from __future__ import annotations

import numpy as np

from ..hamiltonian.ewald import ewald
from ..hamiltonian.hartree import hartree
from ..hamiltonian.local import structure_factors
from ..hamiltonian.system import PlaneWaveSystem
from ..pseudo.formfactors import (
    density_form_factor_derivative,
    local_potential_form_factor,
    local_potential_form_factor_derivative,
    projector_form_factor,
    projector_form_factor_derivative,
)
from ..pseudo.kb import real_spherical_harmonics, solid_harmonics_gradient
from ..system.symmetry import symmetrize_tensor

__all__ = ["StressContributions", "compute_stress", "pulay_stress_estimate"]


class StressContributions(dict):
    """The stress decomposition in Ha/Bohr^3. Summing the values gives the total."""

    @property
    def total(self) -> np.ndarray:
        return sum(self.values())


def compute_stress(
    system: PlaneWaveSystem,
    density: np.ndarray,
    wavefunctions,
    occupations: np.ndarray,
    *,
    eigenvalues: np.ndarray | None = None,
    symmetrize: bool = True,
    decompose: bool = False,
) -> np.ndarray | StressContributions:
    """Stress tensor in Ha/Bohr^3, shape ``(3, 3)``.

    ``decompose=True`` returns the individual contributions, which is what the
    finite-difference tests check term by term. ``eigenvalues`` is required for PAW and
    ignored otherwise; see :func:`_paw_stress`.
    """
    if system.is_paw:
        contributions = _paw_stress(system, density, wavefunctions, occupations, eigenvalues)
        if symmetrize and system.symmetry is not None:
            for name, value in contributions.items():
                contributions[name] = symmetrize_tensor(value, system.symmetry, system.cell)
        return contributions if decompose else contributions.total

    grid, sphere = system.grid, system.density_sphere
    total_density = density.sum(axis=0)
    density_g = grid.gather(grid.to_reciprocal(total_density.astype(complex)), sphere)

    contributions = StressContributions()
    charges = np.array([system.pseudopotentials[s].valence_charge for s in system.symbols])
    contributions["ewald"] = ewald(system.cell, system.positions, charges).stress
    contributions["kinetic"] = _kinetic_stress(system, wavefunctions, occupations)
    contributions["hartree"] = _hartree_stress(system, total_density, density_g)
    contributions["local"] = _local_stress(system, density_g)
    contributions["nonlocal"] = _nonlocal_stress(system, wavefunctions, occupations)

    xc_density = density
    if system.core_density is not None:
        xc_density = density + system.core_density[None, ...] / system.nspin
    xc = system.xc.evaluate_on_grid(xc_density, grid)
    contributions["xc"] = _xc_stress(system, xc_density, xc)
    if system.core_density is not None:
        contributions["core"] = _core_stress(system, xc)

    if symmetrize and system.symmetry is not None:
        for name, value in contributions.items():
            contributions[name] = symmetrize_tensor(value, system.symmetry, system.cell)
    return contributions if decompose else contributions.total


# ------------------------------------------------------------------------- terms
def _kinetic_stress(system, wavefunctions, occupations) -> np.ndarray:
    """``-(1/Omega) sum w f |c|^2 (k+G)_u (k+G)_v``: the basis functions stretch with the cell."""
    degeneracy = 2.0 / system.nspin
    stress = np.zeros((3, 3))
    for spin in range(system.nspin):
        for ik, sphere in enumerate(system.wave_spheres):
            weights = degeneracy * system.kpoints.weights[ik] * occupations[spin, ik]
            amplitude = np.einsum("n,ng->g", weights, np.abs(wavefunctions[spin][ik]) ** 2)
            stress -= np.einsum("g,gu,gv->uv", amplitude, sphere.cartesian, sphere.cartesian)
    return stress / system.grid.volume


def _hartree_stress(system, total_density, density_g) -> np.ndarray:
    """``-E_H delta/Omega + sum_G 4 pi |n(G)|^2 G_u G_v / G^4``."""
    energy, _ = hartree(system.grid, system.density_sphere, total_density)
    sphere = system.density_sphere
    nonzero = sphere.g2 > 1e-12
    g2 = sphere.g2[nonzero]
    weight = 4.0 * np.pi * np.abs(density_g[nonzero]) ** 2 / g2**2
    vectors = sphere.cartesian[nonzero]
    return -np.eye(3) * energy / system.grid.volume + np.einsum(
        "g,gu,gv->uv", weight, vectors, vectors
    )


def _local_stress(system, density_g) -> np.ndarray:
    """Volume dilution of ``E_loc`` plus the stretch of each ``v~_a(|G|)``."""
    sphere = system.density_sphere
    q = np.sqrt(sphere.g2)
    phases = structure_factors(sphere, system.positions)
    energy = 0.0
    tensor = np.zeros((3, 3))
    nonzero = q > 1e-12
    ratio = np.zeros_like(q)
    ratio[nonzero] = 1.0 / q[nonzero]
    for atom, symbol in enumerate(system.symbols):
        pseudo = system.pseudopotentials[symbol]
        form = local_potential_form_factor(pseudo, q)
        slope = local_potential_form_factor_derivative(pseudo, q)
        energy += float(np.sum((form * phases[atom].conj() * density_g).real))
        weight = (slope * ratio * phases[atom].conj() * density_g).real
        tensor -= np.einsum("g,gu,gv->uv", weight, sphere.cartesian, sphere.cartesian)
    return (-np.eye(3) * energy + tensor) / system.grid.volume


def _xc_stress(system, xc_density, xc) -> np.ndarray:
    """``delta_uv (E_xc - sum_s int v_xc^s n_s)`` plus the GGA gradient outer product.

    ``v_xc^s`` here is the full functional derivative, divergence term included; that is
    what makes the ``delta_uv`` bracket collapse to this form, because integrating the
    divergence by parts contributes exactly the ``2 vsigma sigma`` pieces the strain
    derivative of ``sigma`` produces.
    """
    grid = system.grid
    weight = grid.cell_volume_per_point
    double_counting = float(np.sum(xc.potential * xc_density)) * weight
    stress = np.eye(3) * (xc.energy - double_counting)

    if xc.vsigma is not None:
        gradients, vsigma = xc.gradients, xc.vsigma

        def contract(weight, left, right):
            return np.einsum("ijk,ijku,ijkv->uv", weight, left, right)

        outer = (
            2.0 * contract(vsigma[0], gradients[0], gradients[0])
            + contract(vsigma[1], gradients[0], gradients[1])
            + contract(vsigma[1], gradients[1], gradients[0])
            + 2.0 * contract(vsigma[2], gradients[1], gradients[1])
        )
        stress -= outer * weight
    return stress / grid.volume


def _core_stress(system, xc) -> np.ndarray:
    """The partial core is rigid, not diluted: it stretches in ``|G|`` like the potential.

    The ``n -> n/det`` scaling assumed for the whole exchange-correlation density is right
    for the valence electrons and wrong for the frozen core, which merely rides along with
    its atom. This term is the difference between the two behaviours.
    """
    grid, sphere = system.grid, system.density_sphere
    # The core is split equally between spin channels, so the conjugate potential is the
    # channel average of v_xc.
    average = xc.potential.mean(axis=0)
    potential_g = grid.gather(grid.to_reciprocal(average.astype(complex)), sphere)
    q = np.sqrt(sphere.g2)
    nonzero = q > 1e-12
    ratio = np.zeros_like(q)
    ratio[nonzero] = 1.0 / q[nonzero]
    phases = structure_factors(sphere, system.positions)

    tensor = np.zeros((3, 3))
    for atom, symbol in enumerate(system.symbols):
        pseudo = system.pseudopotentials[symbol]
        if pseudo.core_density is None:
            continue
        slope = density_form_factor_derivative(pseudo.grid, pseudo.core_density, q)
        weight = (potential_g.conj() * slope * ratio * phases[atom]).real
        tensor -= np.einsum("g,gu,gv->uv", weight, sphere.cartesian, sphere.cartesian)
    return tensor / grid.volume


def _nonlocal_stress(system, wavefunctions, occupations) -> np.ndarray:
    """Strain response of the Kleinman-Bylander projectors.

    Two pieces: the ``1/sqrt(Omega)`` normalisation dilutes like everything else, and the
    projector itself is a function of ``k+G``, which the strain stretches and rotates.
    The second needs ``dPhi/dq`` -- both the radial derivative ``f_l'(q)`` and the angular
    one through the spherical harmonic, which is why the solid harmonics and their
    gradients exist in :mod:`amaterasu.pseudo.kb`.
    """
    degeneracy = 2.0 / system.nspin
    stress = np.zeros((3, 3))
    energy = 0.0
    for ik, sphere in enumerate(system.wave_spheres):
        projectors = system.projectors[ik]
        if len(projectors) == 0:
            continue
        gradient = _projector_q_gradient(system, sphere)  # (3, nproj, npw)
        q = sphere.cartesian
        for spin in range(system.nspin):
            weights = degeneracy * system.kpoints.weights[ik] * occupations[spin, ik]
            active = weights > 1e-12
            if not np.any(active):
                continue
            block = wavefunctions[spin][ik][active]
            weights = weights[active]
            projected = projectors.project(block)
            coupled = projected @ projectors.dij
            energy += float(np.einsum("n,ni,ni->", weights, coupled, projected.conj()).real)
            for mu in range(3):
                # R_i^(mu nu) = sum_G conj(dP_i/dq_mu) q_nu c_G
                partial = np.einsum("ng,ig,gv->niv", block, gradient[mu].conj(), q)
                stress[mu] -= np.einsum("n,ni,niv->v", weights, coupled.conj(), partial).real
    # The strain derivative symmetrises as R^(uv) + R^(vu), which is the transpose sum --
    # not its average: both orderings appear in the derivative of a symmetric strain.
    stress = stress + stress.T
    return (stress - np.eye(3) * energy) / system.grid.volume


def _projector_q_gradient(system: PlaneWaveSystem, sphere) -> np.ndarray:
    """``dP_i/dq_alpha`` for every projector at this k-point, shape ``(3, nproj, npw)``.

    ``Phi = 4 pi (-i)^l Y_lm(q_hat) f_l(q)`` with ``Y_lm = C_lm / q^l``, so

        dPhi/dq_a = 4 pi (-i)^l [ (dC/dq_a) f_l / q^l
                                  - l Y_lm f_l q_a / q^2
                                  + Y_lm f_l' q_a / q ] .
    """
    q_vectors = sphere.cartesian
    q = np.sqrt(sphere.g2)
    nonzero = q > 1e-12
    inverse = np.zeros_like(q)
    inverse[nonzero] = 1.0 / q[nonzero]
    prefactor = 4.0 * np.pi / np.sqrt(system.grid.volume)

    rows: list[np.ndarray] = []
    for atom, symbol in enumerate(system.symbols):
        pseudo = system.pseudopotentials[symbol]
        phase = np.exp(-1j * (q_vectors @ system.positions[atom]))
        for projector in pseudo.projectors:
            l = projector.l
            radial = projector_form_factor(pseudo.grid, projector, q)
            slope = projector_form_factor_derivative(pseudo.grid, projector, q)
            harmonics = real_spherical_harmonics(l, q_vectors)
            gradients = solid_harmonics_gradient(l, q_vectors)
            power = inverse**l if l else np.ones_like(q)
            for m in range(2 * l + 1):
                angular = (
                    gradients[m] * (radial * power)[:, None]
                    - l * (harmonics[m] * radial * inverse**2)[:, None] * q_vectors
                    + (harmonics[m] * slope * inverse)[:, None] * q_vectors
                )
                rows.append(prefactor * (-1j) ** l * angular * phase[:, None])
    return np.asarray(rows).transpose(2, 0, 1)


def pulay_stress_estimate(
    atoms, config, *, factor: float = 1.3, logger=None
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the basis-set (Pulay) stress by repeating the calculation at a higher cutoff.

    Returns ``(stress_at_ecut, stress_at_factor*ecut)`` in Ha/Bohr^3. The difference is
    the part of the stress that comes from the plane-wave set being incomplete rather than
    from the physics, and it is almost always a spurious *compressive* pressure: raising
    the cutoff adds basis functions, lowers the energy, and does so more at small volumes
    where the reciprocal cell is coarser.

    There is no cheap shortcut. The estimate costs a second self-consistent calculation,
    which is why it is a diagnostic to run once for a system rather than something applied
    at every relaxation step.
    """
    from copy import deepcopy

    from ..calculator import Amaterasu
    from ..units import BOHR_IN_ANG, HARTREE_IN_EV

    scale = HARTREE_IN_EV / BOHR_IN_ANG**3
    stresses = []
    for multiplier in (1.0, factor):
        scaled = deepcopy(config)
        if scaled.basis.ecut is None:
            raise ValueError("basis.ecut must be set explicitly to estimate the Pulay stress")
        from ..units import parse_energy

        scaled.basis.ecut = parse_energy(config.basis.ecut, "Ry") * 2.0 * multiplier
        scaled.basis.ecut_rho = None
        copy = atoms.copy()
        copy.calc = Amaterasu(scaled, logger=logger)
        voigt = copy.get_stress()
        stresses.append(
            np.array(
                [
                    [voigt[0], voigt[5], voigt[4]],
                    [voigt[5], voigt[1], voigt[3]],
                    [voigt[4], voigt[3], voigt[2]],
                ]
            )
            / scale
        )
    return stresses[0], stresses[1]


# --------------------------------------------------------------------------------- PAW
# The PAW stress keeps the kinetic and exchange-correlation terms of the norm-conserving
# case unchanged and replaces the rest. There is no Ewald sum and no local ionic
# pseudopotential; in their place are the zero potential, the smooth core (which is a
# charge here, not only an exchange-correlation device) and the compensation charge, which
# is the only rigid atomic function in the code with an angular dependence and therefore
# the only one whose strain derivative needs the solid harmonics as well as a radial
# derivative. The one-centre terms contribute nothing directly: they live on a radial mesh
# that does not strain, and their dependence on the cell is entirely through the
# occupancies, which is already carried by the projector term.


def _paw_stress(system, density, wavefunctions, occupations, eigenvalues):
    from ..scf.driver import compute_occupancies, effective_potential, paw_state

    if eigenvalues is None:
        raise ValueError(
            "the PAW stress needs the eigenvalues: the overlap operator strains with the "
            "cell, so the orthonormality constraint contributes a term weighted by them"
        )
    grid, sphere = system.grid, system.density_sphere
    occupancies = compute_occupancies(system, wavefunctions, occupations)
    state = paw_state(system, occupancies)
    potential, _, _, hartree_g = effective_potential(system, density, state)
    coupling = system.paw.coupling(state.onsite, hartree_g, system.nspin)

    smooth = density.sum(axis=0)
    if system.core_density is not None:
        smooth = smooth + system.core_density
    compensated = smooth + state.compensation
    smooth_g = grid.gather(grid.to_reciprocal(smooth.astype(complex)), sphere)
    density_g = grid.gather(grid.to_reciprocal(compensated.astype(complex)), sphere)

    contributions = StressContributions()
    contributions["kinetic"] = _kinetic_stress(system, wavefunctions, occupations)
    contributions["hartree"] = _hartree_stress(system, compensated, density_g)

    xc_density = density
    if system.core_density is not None:
        xc_density = density + system.core_density[None, ...] / system.nspin
    xc = system.xc.evaluate_on_grid(xc_density, grid)
    contributions["xc"] = _xc_stress(system, xc_density, xc)
    contributions["core"] = _paw_core_stress(system, potential.mean(axis=0))
    contributions["vbar"] = _vbar_stress(system, smooth_g)
    contributions["compensation"] = _compensation_stress(system, state, hartree_g)
    contributions["nonlocal"] = _paw_nonlocal_stress(
        system, wavefunctions, occupations, eigenvalues, coupling
    )
    return contributions


def _paw_core_stress(system, mean_potential) -> np.ndarray:
    """The smooth core is rigid, not diluted; its conjugate potential is the whole of v_eff.

    Unlike a norm-conserving partial core, which only enters exchange-correlation, the PAW
    smooth core is a charge in the Hartree term and a partner of the zero potential as
    well. All three are contained in ``v_eff``, averaged over spin because the core is not
    polarised.
    """
    grid, sphere = system.grid, system.density_sphere
    potential_g = grid.gather(grid.to_reciprocal(mean_potential.astype(complex)), sphere)
    q = np.sqrt(sphere.g2)
    inverse = np.zeros_like(q)
    nonzero = q > 1e-12
    inverse[nonzero] = 1.0 / q[nonzero]
    phases = structure_factors(sphere, system.positions)

    tensor = np.zeros((3, 3))
    for atom, symbol in enumerate(system.symbols):
        data = system.paw.setups[symbol].dataset
        slope = density_form_factor_derivative(data.grid, data.core_density, q)
        weight = (potential_g.conj() * slope * inverse * phases[atom]).real
        tensor -= np.einsum("g,gu,gv->uv", weight, sphere.cartesian, sphere.cartesian)
    return tensor / grid.volume


def _vbar_stress(system, smooth_g) -> np.ndarray:
    """Volume dilution of the zero-potential energy plus the stretch of each ``v_bar(|G|)``."""
    from ..pseudo.formfactors import radial_transform, radial_transform_derivative

    sphere = system.density_sphere
    q = np.sqrt(sphere.g2)
    inverse = np.zeros_like(q)
    nonzero = q > 1e-12
    inverse[nonzero] = 1.0 / q[nonzero]
    phases = structure_factors(sphere, system.positions)

    energy = 0.0
    tensor = np.zeros((3, 3))
    for atom, symbol in enumerate(system.symbols):
        setup = system.paw.setups[symbol]
        data = setup.dataset
        values = np.zeros(len(data.grid))
        stop = setup.stop
        values[:stop] = np.sqrt(4.0 * np.pi) * data.zero_potential[:stop] * data.grid.r[:stop] ** 2
        form = radial_transform(data.grid, values, 0, q)
        slope = radial_transform_derivative(data.grid, values, 0, q)
        energy += float(np.sum((form * phases[atom].conj() * smooth_g).real))
        weight = (slope * inverse * phases[atom].conj() * smooth_g).real
        tensor -= np.einsum("g,gu,gv->uv", weight, sphere.cartesian, sphere.cartesian)
    return (-np.eye(3) * energy + tensor) / system.grid.volume


def _compensation_stress(system, state, hartree_g) -> np.ndarray:
    """Strain response of the compensation charge, whose form factor is not spherical.

    ``Z~^a(G) = (1/Omega) sum_L Q_L 4 pi (-i)^l h(q) C_L(G) exp(-i G.R_a)`` with
    ``h = g~_l/q^l`` and ``C_L`` the solid harmonic, so

        d/dG_u [h C_L] = (h'/q) G_u C_L + h dC_L/dG_u ,

    and for a Gaussian ``h'/q = -h/(2 alpha)`` -- no ``1/q`` anywhere, which is what makes
    this term as well conditioned as the spherical ones. The volume dilution of the energy
    is already in the Hartree term, which sees ``Z~`` as part of the charge density.
    """
    from ..pseudo.paw.angular import solid_harmonics_any_l, solid_harmonics_gradient_any_l

    sphere = system.density_sphere
    gvectors = sphere.cartesian
    q = np.sqrt(sphere.g2)
    phases = structure_factors(sphere, system.positions)

    tensor = np.zeros((3, 3))
    for atom, symbol in enumerate(system.symbols):
        setup = system.paw.setups[symbol]
        shape = setup.shape
        moments = state.onsite[atom].multipoles
        for l in range(setup.lmax_density + 1):
            harmonics = solid_harmonics_any_l(l, gvectors)
            gradients = solid_harmonics_gradient_any_l(l, gvectors)
            envelope = shape.envelope(l, q)
            slope = shape.envelope_slope_over_q(l, q)
            for m in range(2 * l + 1):
                index = l * l + m
                if abs(moments[index]) < 1e-30:
                    continue
                base = (
                    hartree_g.conj() * (4.0 * np.pi * (-1j) ** l * moments[index]) * phases[atom]
                ).real
                gradient = (slope * harmonics[m])[:, None] * gvectors + envelope[
                    :, None
                ] * gradients[m]
                tensor -= np.einsum("g,gu,gv->uv", base, gradient, gvectors)
    return tensor / system.grid.volume


def _paw_projector_q_gradient(system, sphere) -> np.ndarray:
    """``dP_i/dq_alpha`` for the PAW projectors, shape ``(3, nproj, npw)``."""
    from ..pseudo.formfactors import radial_transform, radial_transform_derivative
    from ..pseudo.paw.angular import (
        real_harmonics,
        solid_harmonics_gradient_any_l,
    )

    gvectors = sphere.cartesian
    q = np.sqrt(sphere.g2)
    nonzero = q > 1e-12
    inverse = np.zeros_like(q)
    inverse[nonzero] = 1.0 / q[nonzero]
    prefactor = 4.0 * np.pi / np.sqrt(system.grid.volume)

    rows: list[np.ndarray] = []
    for atom, symbol in enumerate(system.symbols):
        setup = system.paw.setups[symbol]
        data = setup.dataset
        phase = np.exp(-1j * (gvectors @ system.positions[atom]))
        for wave in data.waves:
            l = wave.l
            weight = np.zeros(len(data.grid))
            weight[: setup.stop] = wave.projector[: setup.stop] * data.grid.r[: setup.stop] ** 2
            radial = radial_transform(data.grid, weight, l, q)
            slope = radial_transform_derivative(data.grid, weight, l, q)
            harmonics = real_harmonics(l, gvectors)[l**2 :]
            gradients = solid_harmonics_gradient_any_l(l, gvectors)
            power = inverse**l if l else np.ones_like(q)
            for m in range(2 * l + 1):
                angular = (
                    gradients[m] * (radial * power)[:, None]
                    - l * (harmonics[m] * radial * inverse**2)[:, None] * gvectors
                    + (harmonics[m] * slope * inverse)[:, None] * gvectors
                )
                rows.append(prefactor * (-1j) ** l * angular * phase[:, None])
    return np.asarray(rows).transpose(2, 0, 1)


def _paw_nonlocal_stress(system, wavefunctions, occupations, eigenvalues, coupling):
    """Strain response of the projectors, with the orthonormality constraint included.

    The same structure as the norm-conserving term, with ``D_ij`` replaced by the PAW
    coupling and an extra ``-eps_n O_ij``: the overlap operator is built from the same
    projectors, so straining the cell changes the constraint the orbitals satisfy as well
    as the energy they carry.
    """
    degeneracy = 2.0 / system.nspin
    overlap = system.paw.overlap
    stress = np.zeros((3, 3))
    energy = 0.0
    for ik, sphere in enumerate(system.wave_spheres):
        projectors = system.projectors[ik]
        if len(projectors) == 0:  # pragma: no cover - a cell with no projectors
            continue
        gradient = _paw_projector_q_gradient(system, sphere)
        q_vectors = sphere.cartesian
        for spin in range(system.nspin):
            weights = degeneracy * system.kpoints.weights[ik] * occupations[spin, ik]
            active = weights > 1e-12
            if not np.any(active):
                continue
            block = wavefunctions[spin][ik][active]
            levels = eigenvalues[spin, ik][active]
            weights = weights[active]
            projected = projectors.project(block)
            coupled = projected @ coupling[spin] - levels[:, None] * (projected @ overlap)
            energy += float(np.einsum("n,ni,ni->", weights, coupled, projected.conj()).real)
            for mu in range(3):
                partial = np.einsum("ng,ig,gv->niv", block, gradient[mu].conj(), q_vectors)
                stress[mu] -= np.einsum("n,ni,niv->v", weights, coupled.conj(), partial).real
    stress = stress + stress.T
    return (stress - np.eye(3) * energy) / system.grid.volume
