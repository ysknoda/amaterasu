"""Reading a Kresse-Joubert PAW dataset, and the two quantities that have to be derived.

A pslibrary ``*-kjpaw`` UPF file is the only route to PBEsol + PAW, and it is written in a
different formulation from the one the rest of the code uses. Two things it does not carry
have to be reconstructed, and both are tested here rather than taken on trust:

``v_bar``
    Inverted from the local ionic pseudopotential. The synthetic file these tests read is
    produced by running the same transformation *forwards* on a dataset whose ``v_bar`` is
    known, so reading it back is a round trip that has to return what went in. That catches
    every unit, ``Y_00`` and ``r f(r)`` convention error, which is what actually goes wrong
    when reading an unfamiliar format.
``kinetic_energy_differences``
    Integrated from the partial waves. Checked two ways: against a second, independent
    discretisation of the same integral, and against the values a PAW-XML file ships.
"""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.pseudo.paw.dataset import PAWDataset
from amaterasu.pseudo.paw.kresse_joubert import ROOT_FOUR_PI, radial_xc_potential
from amaterasu.pseudo.paw.radial import derivative, integrate
from amaterasu.pseudo.readers import read_pseudopotential
from amaterasu.pseudo.readers.upfpaw import kinetic_energy_differences, read_upf_paw
from amaterasu.xc.driver import XCFunctional


# ------------------------------------------------------------- the exchange-correlation potential
@pytest.mark.parametrize("functional", ["LDA-PW", "PBE", "PBEsol"])
def test_radial_xc_potential_is_the_functional_derivative(functional, synthetic_paw):
    """``v_xc = dE_xc/dn``, checked against a finite difference in the density.

    The conversion needs a *potential*, not a matrix element, so the gradient term is
    written as an explicit divergence rather than integrated by parts. That is a different
    expression from the one the on-site code uses and needs its own check; a wrong
    divergence would be invisible in the energy and wrong in ``v_bar``.
    """
    xc = XCFunctional(functional)
    grid = synthetic_paw.grid
    r = grid.r
    stop = synthetic_paw.sphere_points
    density = synthetic_paw.pseudo_valence_density + synthetic_paw.pseudo_core_density
    # A localised, smooth perturbation that stays well inside the mesh.
    direction = np.where(r < synthetic_paw.rcut, (1.0 - (r / synthetic_paw.rcut) ** 2) ** 4, 0.0)

    def energy(scale: float) -> float:
        n = (density + scale * direction) / ROOT_FOUR_PI
        half = n / 2.0
        rho = np.stack([half, half], axis=-1)
        if xc.is_gga:
            slope = derivative(grid, half)
            sigma = np.stack([slope**2] * 3, axis=-1)
            eps, *_ = xc.evaluate(rho, sigma)
        else:
            eps, _ = xc.evaluate(rho)
        return integrate(grid, np.asarray(eps).ravel() * n * 4.0 * np.pi * r**2, upto=stop)

    step = 1e-5
    numerical = (energy(step) - energy(-step)) / (2.0 * step)
    potential = radial_xc_potential(xc, grid, density) / ROOT_FOUR_PI
    analytic = integrate(grid, potential * direction / ROOT_FOUR_PI * 4.0 * np.pi * r**2, upto=stop)
    assert analytic == pytest.approx(numerical, rel=2e-6)


# ------------------------------------------------------------------ kinetic energy differences
def test_kinetic_differences_agree_with_a_second_discretisation(synthetic_paw):
    """The integrated-by-parts form against the one with a second derivative.

    They discretise the same integral differently -- one differentiates once, the other
    twice -- so agreement is evidence about the quadrature rather than about the algebra.
    Measured: 3.2e-5 Hartree here, on a deliberately coarse 400-point mesh; 8.7e-6 on the
    real JTH silicon table and 5.6e-6 on copper, whose meshes carry 2001 points.
    """
    data = synthetic_paw
    stop = data.sphere_points
    grid = data.grid
    r = grid.r[:stop]
    inverse = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
    angular = data.wave_l

    parts = kinetic_energy_differences(grid, data.stacked("ae"), data.stacked("ps"), angular, stop)
    second = np.zeros_like(parts)
    for which, sign in (("ae", 1.0), ("ps", -1.0)):
        u = data.stacked(which)[:, :stop] * r
        curvature = derivative(grid, derivative(grid, u))
        for i in range(data.n_waves):
            for j in range(data.n_waves):
                if angular[i] != angular[j]:
                    continue
                l = int(angular[i])
                integrand = u[i] * (-0.5 * curvature[j] + 0.5 * l * (l + 1) * u[j] * inverse**2)
                second[i, j] += sign * integrate(grid, integrand, upto=stop)
    assert parts == pytest.approx(second, abs=1e-4)


def test_kinetic_differences_reproduce_the_ones_a_paw_xml_file_ships(synthetic_paw):
    """The synthetic file stores its own, computed with a lower-order derivative.

    Agreement here is what says the reconstruction is the same quantity, not merely a
    self-consistent one. A real ATOMPAW-generated table agrees less well -- 0.3% of the
    largest element for silicon, 1% for copper -- and since two independent
    discretisations here agree with each other to 3e-5, that difference is in how the
    generator derived its numbers rather than in this quadrature.
    """
    data = synthetic_paw
    mine = kinetic_energy_differences(
        data.grid, data.stacked("ae"), data.stacked("ps"), data.wave_l, data.sphere_points
    )
    assert mine == pytest.approx(data.kinetic_differences, abs=2e-4)


# ------------------------------------------------------------------------------ the reader
def test_the_kresse_joubert_round_trip_returns_what_went_in(synthetic_upf_paw_file, synthetic_paw):
    """Everything the two formulations share has to come back unchanged."""
    back = read_upf_paw(synthetic_upf_paw_file)
    assert isinstance(back, PAWDataset)
    assert back.valence_charge == pytest.approx(synthetic_paw.valence_charge)
    assert back.core_electrons == pytest.approx(synthetic_paw.core_electrons)
    # A UPF file locates the sphere by mesh index, not by a stored radius, so what comes
    # back is the last grid point inside r_c rather than r_c itself.
    index = int(np.searchsorted(synthetic_paw.grid.r, synthetic_paw.rcut, side="right")) - 1
    assert back.rcut == pytest.approx(synthetic_paw.grid.r[index], rel=1e-12)
    assert back.rcut <= synthetic_paw.rcut
    assert back.grid.r == pytest.approx(synthetic_paw.grid.r, rel=1e-12)
    assert back.wave_l.tolist() == synthetic_paw.wave_l.tolist()
    for mine, theirs in zip(back.waves, synthetic_paw.waves, strict=True):
        assert mine.occupation == pytest.approx(theirs.occupation)
        # The origin is extrapolated when dividing r f(r) by r, so skip the first point.
        assert mine.ae[1:] == pytest.approx(theirs.ae[1:], rel=1e-11)
        assert mine.ps[1:] == pytest.approx(theirs.ps[1:], rel=1e-11)
        assert mine.projector[1:] == pytest.approx(theirs.projector[1:], rel=1e-11)
    for name in ("ae_core_density", "pseudo_core_density"):
        assert getattr(back, name) == pytest.approx(getattr(synthetic_paw, name), rel=1e-11)
    assert back.pseudo_valence_density[1:] == pytest.approx(
        synthetic_paw.pseudo_valence_density[1:], rel=1e-10
    )


def test_the_zero_potential_survives_the_round_trip(synthetic_upf_paw_file, synthetic_paw):
    """The one quantity a Kresse-Joubert file does not carry, recovered from the one it does.

    This is the whole point of :mod:`amaterasu.pseudo.paw.kresse_joubert`: the file holds
    ``v_H[n~_Zc]``, the transformation is inverted on load, and what comes out has to be the
    ``v_bar`` the dataset was built with.
    """
    back = read_upf_paw(synthetic_upf_paw_file)
    inside = back.grid.r <= back.rcut
    expected = synthetic_paw.zero_potential[inside]
    assert back.zero_potential[inside] == pytest.approx(expected, abs=2e-6)
    # v_bar is short ranged: outside the sphere there is nothing left of it.
    assert np.all(back.zero_potential[~inside] == 0.0)


def test_the_dataset_keeps_its_provenance(synthetic_upf_paw_file):
    back = read_upf_paw(synthetic_upf_paw_file)
    assert back.extra["format"] == "upf-paw"
    assert back.extra["formulation"] == "kresse-joubert"
    assert "kj_local_potential" in back.extra
    assert back.kind == "PAW"


def test_a_upf_file_is_dispatched_by_its_own_header(synthetic_upf_paw_file, si_pseudo):
    """``read_pseudopotential`` has to pick the PAW reader without being told."""
    assert isinstance(read_pseudopotential(synthetic_upf_paw_file), PAWDataset)
    assert not isinstance(si_pseudo, PAWDataset)


def test_a_norm_conserving_upf_is_refused_by_the_paw_reader(tmp_path, synthetic_upf_paw_file):
    text = synthetic_upf_paw_file.read_text(encoding="utf-8").replace(
        'pseudo_type="PAW"', 'pseudo_type="NC"'
    )
    path = tmp_path / "not-paw.UPF"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="not PAW"):
        read_upf_paw(path)
