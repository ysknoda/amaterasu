"""The one-centre PAW correction and the identity that makes it trustworthy.

The Hamiltonian coefficients are *defined* as ``dE/dD_ij``. Checking them against a finite
difference of the energy is therefore not a comparison against an external reference but
against the definition itself, and it is the decisive test for this module: the on-site
energy and its gradient are the two most intricate expressions in PAW, and if they agree,
the self-consistent solution really is the stationary point of the energy being reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.pseudo.paw.onsite import OnsiteSetup
from amaterasu.pseudo.paw.shape import GaussianShape
from amaterasu.xc.driver import XCFunctional


def build(dataset, functional="PBE", **kwargs) -> OnsiteSetup:
    return OnsiteSetup(
        dataset, XCFunctional(functional), GaussianShape.for_radius(dataset.rcut), **kwargs
    )


def occupancy(setup, nspin=1, seed=3, states=3):
    """``D = sum_n f_n c_n c_n^T``: positive semi-definite, as a real occupancy matrix is.

    Positive semi-definiteness is not decoration. It is what keeps the one-centre density
    non-negative, ``n^a = sum_n f_n (sum_i c_ni phi_i)^2``, and a density that goes
    negative sends any exchange-correlation functional to nonsense -- so a random
    symmetric matrix is not a valid test input, however tempting.
    """
    rng = np.random.default_rng(seed)
    size = setup.n_projectors
    damping = np.array([1.0 if setup.dataset.waves[c].is_bound else 0.15 for c in setup.channel_of])
    matrix = np.zeros((nspin, size, size))
    for spin in range(nspin):
        for weight in list(np.linspace(1.2, 0.4, states)):
            vector = rng.normal(size=size) * damping
            matrix[spin] += (weight / nspin) * np.outer(vector, vector)
    return matrix


def spherical_occupancy(setup, nspin=1):
    """The reference atom's filling: each shell equally occupied over its ``m`` values."""
    matrix = np.zeros((nspin, setup.n_projectors, setup.n_projectors))
    for row, channel in enumerate(setup.channel_of):
        wave = setup.dataset.waves[channel]
        if wave.occupation:
            matrix[:, row, row] = wave.occupation / (2 * wave.l + 1) / nspin
    return matrix


@pytest.mark.parametrize("functional", ["LDA-PW", "PBE"])
@pytest.mark.parametrize("nspin", [1, 2])
def test_hamiltonian_is_the_gradient_of_the_energy(synthetic_paw, functional, nspin):
    setup = build(synthetic_paw, functional)
    matrix = occupancy(setup, nspin)
    result = setup.compute(matrix)
    step = 1e-6
    worst = 0.0
    for spin in range(nspin):
        for i in range(setup.n_projectors):
            for j in range(i, setup.n_projectors):
                raised, lowered = matrix.copy(), matrix.copy()
                raised[spin, i, j] += step
                lowered[spin, i, j] -= step
                difference = (setup.compute(raised).energy - setup.compute(lowered).energy) / (
                    2 * step
                )
                worst = max(worst, abs(difference - result.hamiltonian[spin, i, j]))
    assert worst < 1e-5 * max(float(np.abs(result.hamiltonian).max()), 1.0)


@pytest.mark.parametrize("term", ["kinetic", "electrostatic", "xc", "vbar"])
def test_every_term_of_the_gradient_separately(synthetic_paw, term):
    """Term by term, so a discrepancy names one expression instead of their sum.

    The electrostatic entry is the one that caught a real bug: the discrete radial Poisson
    operator is accurate but not self-adjoint on the mesh, which leaves the energy right
    and this gradient several per cent wrong.
    """
    setup = build(synthetic_paw, "PBE")
    matrix = occupancy(setup, 1)
    result = setup.compute(matrix)
    analytic = result.gradient_terms[term]
    analytic = analytic[0] if analytic.ndim == 3 else analytic
    step = 1e-6
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (2, 4)]:
        raised, lowered = matrix.copy(), matrix.copy()
        raised[0, i, j] += step
        lowered[0, i, j] -= step
        difference = (
            setup.compute(raised).decomposition[term] - setup.compute(lowered).decomposition[term]
        ) / (2 * step)
        assert difference == pytest.approx(analytic[i, j], abs=1e-5 * max(abs(analytic[i, j]), 1.0))


def test_decomposition_sums_to_the_energy(synthetic_paw):
    setup = build(synthetic_paw)
    result = setup.compute(occupancy(setup))
    assert sum(result.decomposition.values()) == pytest.approx(result.energy, rel=1e-13)


def test_spin_channels_describe_the_same_state(synthetic_paw):
    """An unpolarised state written with one channel or with two must give one answer."""
    setup = build(synthetic_paw, "PBE")
    single = spherical_occupancy(setup, 1)
    doubled = np.stack([single[0] / 2, single[0] / 2])
    assert setup.compute(single).energy == pytest.approx(setup.compute(doubled).energy, abs=1e-12)


def test_spherical_occupancy_has_no_higher_multipoles(synthetic_paw):
    """A filled shell is isotropic, so every Q_L with l > 0 must vanish exactly.

    This is a direct check on the Gaunt coefficients and the m ordering: get either wrong
    and a spherical atom acquires a spurious dipole or quadrupole.
    """
    setup = build(synthetic_paw)
    moments = setup.multipoles(spherical_occupancy(setup))
    assert np.abs(moments[1:]).max() < 1e-12
    assert moments[0] != 0.0


def test_monopole_accounts_for_the_nuclear_and_core_charge(synthetic_paw):
    """sqrt(4 pi) Q_00 = sum_ij D_ij O_ij - Z + N_core - N_smooth_core, Eqs. (14) and (16).

    Reproducing this by an independent route is what makes the compensated cell exactly
    neutral, and neutrality is what lets the plane-wave Hartree term stand in for the
    whole of the electrostatics, Ewald sum included.
    """
    setup = build(synthetic_paw)
    matrix = occupancy(setup)
    moments = setup.multipoles(matrix)
    grid = synthetic_paw.grid
    stop = synthetic_paw.sphere_points
    from amaterasu.pseudo.paw.radial import integrate

    core_difference = np.sqrt(4.0 * np.pi) * integrate(
        grid,
        (synthetic_paw.ae_core_density - synthetic_paw.pseudo_core_density)[:stop]
        * grid.r[:stop] ** 2,
        upto=stop,
    )
    expected = (
        float(np.einsum("sij,ij->", matrix, setup.overlap))
        - synthetic_paw.atomic_number
        + core_difference
    )
    assert np.sqrt(4.0 * np.pi) * moments[0] == pytest.approx(expected, rel=1e-12)


def test_overlap_coefficients_are_the_l_zero_moments(synthetic_paw):
    """O_ij = sqrt(4 pi) Delta_00ij (Eq. 39) -- the coefficients of the PAW overlap operator."""
    setup = build(synthetic_paw)
    # The one-centre integrals run past r_c to enclose the compensation charge, so the
    # radial arrays are setup.stop long, not sphere_points long. The integrand vanishes in
    # between either way -- the partial waves coincide there.
    stop = setup.stop
    from amaterasu.pseudo.paw.radial import integrate

    grid = synthetic_paw.grid
    ae = setup.radial_ae
    ps = setup.radial_ps
    direct = integrate(
        grid,
        (ae[:, None, :] * ae[None, :, :] - ps[:, None, :] * ps[None, :, :]) * grid.r[:stop] ** 2,
        upto=stop,
    )
    same_lm = setup.lm_of[:, None] == setup.lm_of[None, :]
    assert setup.overlap == pytest.approx(direct * same_lm, abs=1e-12)


def test_occupancies_must_have_the_right_shape(synthetic_paw):
    setup = build(synthetic_paw)
    with pytest.raises(ValueError, match="expected"):
        setup.compute(np.zeros((1, 2, 2)))


def test_a_spherical_density_is_integrated_exactly_at_any_angular_degree(synthetic_paw):
    """The angular rule costs nothing when the on-site density is isotropic.

    Only exchange-correlation is integrated pointwise over the sphere; for a filled shell
    its integrand is constant there, so the coarsest rule is already exact. That is why
    the default degree is a worst-case choice rather than a typical one.
    """
    matrix = spherical_occupancy(build(synthetic_paw))
    energies = [
        build(synthetic_paw, "PBE", angular_degree=degree).compute(matrix).energy
        for degree in (8, 20)
    ]
    assert energies[0] == pytest.approx(energies[1], abs=1e-12)


def test_angular_convergence_of_an_aspherical_density(synthetic_paw):
    """A deliberately extreme aspherical density still lands inside the meV budget."""
    matrix = occupancy(build(synthetic_paw), 1)
    coarse, default, fine = (
        build(synthetic_paw, "PBE", angular_degree=degree).compute(matrix).energy
        for degree in (8, 20, 32)
    )
    assert abs(default - fine) < 4e-6  # ~0.1 meV
    assert abs(coarse - fine) > abs(default - fine)


def test_energy_does_not_care_where_the_mesh_stops(synthetic_paw):
    """``E^a - E~^a`` is a sphere-local quantity; pushing the cut outwards must not move it.

    This is the defining property of the one-centre correction. Because ``Z~^a`` reproduces
    the multipoles of ``n^a - Z delta``, and because the partial waves and the two core
    densities agree outside ``r_c``, the integrands beyond the sphere cancel between the
    all-electron and the smooth term. The test earns its keep because those two terms are
    individually enormous and nearly cancel: an inaccuracy that survives the cancellation
    shows up here as a drift and essentially nowhere else. A quadrature that ran its cubic
    panels over the zero-padded tail of the density -- fitting a cubic through the step at
    ``r_c`` -- drifted by 3.8 eV here while every other test in this file still passed.

    The residual is set by the Gaussian's own leak past ``r_c``; see the table in
    :mod:`amaterasu.pseudo.paw.shape`.
    """
    import dataclasses

    shape = GaussianShape.for_radius(synthetic_paw.rcut)
    reference = None
    for extra in (0, 15, 30, 60):
        stop = min(synthetic_paw.sphere_points + extra, len(synthetic_paw.grid.r))
        # Only the integration limit moves; the compensation charge keeps the width the
        # true sphere radius picked, or the shape would be changing along with the cut.
        widened = dataclasses.replace(synthetic_paw, rcut=synthetic_paw.grid.r[stop - 2])
        setup = OnsiteSetup(widened, XCFunctional("PBE"), shape)
        energy = setup.compute(setup.atomic_occupancies(1, 0.0)).energy
        if reference is None:
            reference = energy
        assert energy == pytest.approx(reference, abs=5e-6), (
            f"the one-centre energy moved by {energy - reference:.3e} Ha when the radial "
            f"mesh was extended to r = {widened.grid.r[stop - 1]:.3f} Bohr"
        )
