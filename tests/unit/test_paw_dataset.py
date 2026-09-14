"""Reading a PAW-XML dataset, and the invariants that say it was read correctly."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.special import spherical_jn

from amaterasu.pseudo.paw.shape import (
    BesselShape,
    GaussianShape,
    double_factorial,
    equivalent_alpha_rc2,
    make_shape,
    spherical_jn_zeros,
)
from amaterasu.pseudo.readers.pawxml import radial_grid_from_element, read_pawxml


def test_reads_the_header(synthetic_paw):
    assert synthetic_paw.symbol == "X"
    assert synthetic_paw.atomic_number == 6.0
    assert synthetic_paw.core_electrons == 2.0
    assert synthetic_paw.valence_charge == 4.0
    assert synthetic_paw.xc_functional == "LDA-PW"
    assert synthetic_paw.ecut_hints == {"low": 12.0, "medium": 16.0, "high": 20.0}
    assert synthetic_paw.kind == "PAW"


def test_biorthogonality_is_machine_precision(synthetic_paw):
    """<p~_i|phi~_j> = delta_ij is the sharpest single check on the radial convention.

    PAW-XML stores the plain radial part f(r); UPF stores r f(r) for the same objects.
    Reading one as the other leaves this quantity wrong by orders of magnitude, so a
    tight bound here pins the convention without needing a reference calculation.
    """
    assert synthetic_paw.biorthogonality_error() < 1e-12


def test_core_density_holds_the_declared_number_of_electrons(synthetic_paw):
    """sqrt(4 pi) integral n_c r^2 dr = N_core fixes the Y_00 factor in the convention."""
    assert synthetic_paw.core_charge_error() < 1e-10


def test_partial_waves_coincide_outside_the_augmentation_radius(synthetic_paw):
    """Everything one-centre is a difference that vanishes outside r_c; check it does."""
    assert synthetic_paw.partial_waves_match_outside() < 1e-12


def test_check_passes_on_a_sound_dataset(synthetic_paw):
    assert synthetic_paw.check() == []


def test_sphere_points_bracket_the_augmentation_radius(synthetic_paw):
    stop = synthetic_paw.sphere_points
    radii = synthetic_paw.grid.r
    assert radii[stop - 1] >= synthetic_paw.rcut > radii[stop - 3]


def test_unbound_partial_waves_are_recognised(synthetic_paw):
    bound = [w.is_bound for w in synthetic_paw.waves]
    assert bound == [True, False, True]
    assert synthetic_paw.l_max == 1
    assert list(synthetic_paw.wave_l) == [0, 0, 1]


def test_kinetic_differences_are_symmetric_and_l_block_diagonal(synthetic_paw):
    kinetic = synthetic_paw.kinetic_differences
    assert kinetic == pytest.approx(kinetic.T, abs=1e-12)
    assert abs(kinetic[0, 2]) < 1e-14  # an s channel cannot couple to a p channel


def test_plane_wave_density_convention(synthetic_paw):
    """The grid side wants 4 pi r^2 rho; converting must preserve the enclosed charge."""
    converted = synthetic_paw.as_plane_wave_density("ae_core")
    assert synthetic_paw.grid.integrate(converted) == pytest.approx(2.0, abs=1e-9)


# ------------------------------------------------------------------- radial grids
@pytest.mark.parametrize(
    "attributes",
    [
        'eq="r=d*i" d="0.01"',
        'eq="r=a*exp(d*i)" a="1e-4" d="0.02"',
        'eq="r=a*(exp(d*i)-1)" a="1e-3" d="0.02"',
        'eq="r=a*i/(1-b*i)" a="0.01" b="0.001"',
        'eq="r=a*i/(n-i)" a="0.02" n="600"',
        'eq="r=(i/n+a)^5/a-a^4" a="0.3" n="400"',
    ],
)
def test_every_radial_grid_equation(attributes):
    """All six meshes of the specification, each with dr/di consistent with r(i)."""
    element = ET.fromstring(f'<radial_grid {attributes} istart="0" iend="199" id="g"/>')
    grid = radial_grid_from_element(element)
    assert len(grid) == 200
    assert np.all(np.diff(grid.r) > 0)
    # dr/di must match a central difference of r(i) itself.
    numerical = np.gradient(grid.r)
    assert grid.dr_di[5:-5] == pytest.approx(numerical[5:-5], rel=2e-3)


def test_tabulated_mesh_must_agree_with_the_equation():
    """A file whose <values> disagree with its own equation is misread, not tolerated."""
    element = ET.fromstring(
        '<radial_grid eq="r=d*i" d="0.01" istart="0" iend="3" id="g">'
        "<values>0.0 0.01 0.02 0.99</values></radial_grid>"
    )
    with pytest.raises(ValueError, match="disagrees with the equation"):
        radial_grid_from_element(element, "test")


def test_unknown_grid_equation_is_reported():
    element = ET.fromstring('<radial_grid eq="r=sinh(i)" istart="0" iend="9"/>')
    with pytest.raises(ValueError, match="unsupported radial_grid equation"):
        radial_grid_from_element(element, "test")


def test_wrong_root_element(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text("<not_a_paw_dataset/>", encoding="utf-8")
    with pytest.raises(ValueError, match="expected <paw_dataset>"):
        read_pawxml(path)


def test_missing_kinetic_block_is_reported(tmp_path, synthetic_pawxml_text):
    path = tmp_path / "truncated.xml"
    text = synthetic_pawxml_text.replace(
        synthetic_pawxml_text[synthetic_pawxml_text.index("<kinetic_energy_differences>") :],
        "</paw_dataset>\n",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="kinetic_energy_differences"):
        read_pawxml(path)


# ---------------------------------------------------------------- shape functions
def test_double_factorial():
    assert [double_factorial(l) for l in range(4)] == [1.0, 3.0, 15.0, 105.0]


@pytest.mark.parametrize("l", [0, 1, 2, 3])
@pytest.mark.parametrize("kind", ["gaussian", "bessel"])
def test_shape_function_multipole_normalisation(kind, l, synthetic_paw):
    """integral g_l r^(l+2) dr = 1 is what makes Q_L the actual multipole moment."""
    from scipy.integrate import quad

    shape = make_shape(kind, synthetic_paw.rcut)

    def integrand(r):
        return float(shape.radial(l, np.array(r))) * r ** (l + 2)

    # Split at the containment radius rather than integrating straight to infinity: for a
    # compact shape that is where the integrand stops, and quad cannot find a support it
    # is not told about. The outer piece is not always negligible -- the containment radius
    # is defined from the *monopole*, and a Gaussian's r^l weighting pushes the higher
    # multipoles a little further out, by 2e-9 of the moment at l = 3.
    upper = shape.containment_radius()
    moment = quad(integrand, 0.0, upper, limit=400)[0] + quad(integrand, upper, np.inf)[0]
    assert moment == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_shape_form_factor_matches_the_numerical_transform(l, synthetic_paw):
    """The closed-form Fourier transform is what puts the compensation charge on the grid."""
    from amaterasu.pseudo.formfactors import radial_transform

    shape = GaussianShape.for_radius(synthetic_paw.rcut)
    grid = synthetic_paw.grid
    q = np.array([0.0, 0.4, 1.3, 3.0, 7.0])
    numerical = radial_transform(grid, shape.radial(l, grid.r) * grid.r**2, l, q)
    assert shape.form_factor(l, q) == pytest.approx(numerical, abs=1e-12)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_shape_form_factor_derivative(l):
    shape = GaussianShape(alpha=2.5, rcut=1.9)
    q = np.array([0.7, 2.3, 5.0])
    step = 1e-5
    reference = (shape.form_factor(l, q + step) - shape.form_factor(l, q - step)) / (2 * step)
    assert shape.form_factor_derivative(l, q) == pytest.approx(reference, abs=1e-9)


def test_escaped_charge_matches_direct_integration():
    """How much of the compensation charge falls outside the sphere is the number to watch.

    Checked against an adaptive quadrature rather than the radial mesh: the integrand is
    cut off sharply at r_c, and Simpson's rule across that step is worse than the 1e-4
    quantity being measured.
    """
    from scipy.integrate import quad

    shape = GaussianShape.for_radius(1.9)
    normalisation = shape.normalisation(0)
    escaped = quad(lambda r: normalisation * np.exp(-shape.alpha * r**2) * r**2, 1.9, np.inf)[0]
    assert shape.escaped_charge() == pytest.approx(escaped, rel=1e-8)
    assert shape.escaped_charge() < 1e-3
    # enclosed_monopole is an independent closed form for the complementary quantity.
    assert shape.enclosed_monopole(1.9) == pytest.approx(1.0 - escaped, rel=1e-10)


def test_zero_radius_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        GaussianShape.for_radius(0.0)


# ------------------------------------------------------- compactly supported (bessel)
def test_spherical_jn_zeros_are_the_right_zeros():
    """The interlacing bracket must not let the root finder skip or repeat a zero."""
    assert spherical_jn_zeros(0, 3) == pytest.approx([np.pi, 2 * np.pi, 3 * np.pi], rel=1e-14)
    # j_1's first zero is the tan(x) = x root, 4.493409457909064.
    assert spherical_jn_zeros(1, 1)[0] == pytest.approx(4.493409457909064, rel=1e-13)
    for l in range(5):
        zeros = np.array(spherical_jn_zeros(l, 3))
        assert np.all(np.diff(zeros) > 0)
        assert spherical_jn(l, zeros) == pytest.approx(np.zeros(3), abs=1e-14)


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_bessel_shape_is_exactly_zero_outside_its_radius(l):
    """Compact support is the whole reason this shape exists, so assert it exactly."""
    shape = BesselShape(1.9)
    beyond = np.array([1.9 + 1e-12, 1.95, 2.5, 10.0])
    assert np.all(shape.radial(l, beyond) == 0.0)
    assert shape.containment_radius() == 1.9
    assert shape.escaped_charge() == 0.0


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_bessel_shape_meets_zero_smoothly(l):
    """g(r_c) = 0 and g'(r_c) = 0: the two conditions the amplitudes were solved from.

    Without the second one the shape would have a kink at r_c, and its form factor would
    decay one power of |G| more slowly than it already does.
    """
    shape = BesselShape(1.9)
    assert float(shape.radial(l, np.array(1.9))) == pytest.approx(0.0, abs=1e-15)
    step = 1e-6
    inward = np.array([1.9 - step, 1.9 - 2 * step, 1.9 - 3 * step])
    slope = (
        3 * shape.radial(l, inward[0]) - 4 * shape.radial(l, inward[1]) + shape.radial(l, inward[2])
    ) / (2 * step)
    assert float(slope) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_bessel_form_factor_matches_the_numerical_transform(l):
    """Checked against adaptive quadrature, not the radial mesh.

    The shape stops dead at r_c, and a fixed-stencil rule straddling that step is worse
    than the agreement being measured -- the same reason the escaped-charge test uses quad.
    """
    from scipy.integrate import quad

    shape = BesselShape(1.9094498728)
    for q in [0.0, 0.37, 2.1, 5.5, 11.0, 17.9]:
        reference = quad(
            lambda r, q=q: float(shape.radial(l, np.array(r))) * spherical_jn(l, q * r) * r**2,
            0.0,
            shape.rcut,
            limit=800,
        )[0]
        assert float(shape.form_factor(l, np.array([q]))[0]) == pytest.approx(reference, abs=1e-14)


@pytest.mark.parametrize("l", [0, 2, 4])
def test_bessel_form_factor_survives_its_own_poles(l):
    """The closed form divides by ``z_i^2 - (q r_c)^2``, and a G vector can sit on a zero.

    Written as it stands the quotient is 0/0 exactly there and loses a digit per decade of
    approach, which no energy-only test would show: one Fourier component out of a million
    would be wrong, or nan. The series patch has to hold right through the pole.
    """
    from scipy.integrate import quad

    shape = BesselShape(1.9094498728)
    for zero in spherical_jn_zeros(l, 2):
        pole = zero / shape.rcut
        for offset in [-1e-4, -1e-9, 0.0, 1e-9, 1e-4]:
            q = pole + offset
            reference = quad(
                lambda r, q=q: float(shape.radial(l, np.array(r))) * spherical_jn(l, q * r) * r**2,
                0.0,
                shape.rcut,
                limit=1200,
            )[0]
            value = float(shape.form_factor(l, np.array([q]))[0])
            assert np.isfinite(value)
            assert value == pytest.approx(reference, rel=1e-11, abs=1e-15)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_bessel_envelope_reproduces_the_form_factor_and_its_slope(l):
    """``envelope`` and ``envelope_slope_over_q`` are what the stress tensor differentiates."""
    shape = BesselShape(1.9)
    q = np.array([0.4, 1.7, 3.3, 8.0, 15.0])
    assert shape.form_factor(l, q) == pytest.approx(q**l * shape.envelope(l, q), rel=1e-13)
    step = 1e-5
    reference = (shape.envelope(l, q + step) - shape.envelope(l, q - step)) / (2 * step) / q
    assert shape.envelope_slope_over_q(l, q) == pytest.approx(reference, rel=1e-7)
    slope = (shape.form_factor(l, q + step) - shape.form_factor(l, q - step)) / (2 * step)
    assert shape.form_factor_derivative(l, q) == pytest.approx(slope, rel=1e-7)


def test_bessel_envelope_is_finite_at_the_origin():
    """h(0) = 1/(2l+1)!!, the same small-q limit a Gaussian has: both carry unit moment."""
    shape = BesselShape(1.9)
    for l in range(5):
        assert float(shape.envelope(l, np.array([0.0]))[0]) == pytest.approx(
            1.0 / double_factorial(l), rel=1e-13
        )
        assert np.isfinite(shape.envelope_slope_over_q(l, np.array([0.0]))[0])
    assert float(shape.form_factor(0, np.array([0.0]))[0]) == pytest.approx(1.0, abs=1e-15)


def test_bessel_enclosed_monopole_is_the_closed_form_of_its_own_radial():
    from scipy.integrate import quad

    shape = BesselShape(1.9)
    for r in [0.25, 0.9, 1.6, 1.9, 3.0]:
        reference = quad(
            lambda s: float(shape.radial(0, np.array(s))) * s**2, 0.0, min(r, 1.9), limit=400
        )[0]
        assert float(shape.enclosed_monopole(np.array(r))) == pytest.approx(reference, abs=1e-14)
    assert shape.escaped_charge(1.9 * 0.8) == pytest.approx(0.0204, abs=1e-3)


def test_make_shape_rejects_what_it_cannot_build():
    assert isinstance(make_shape("bessel", 1.9), BesselShape)
    assert isinstance(make_shape("gaussian", 1.9), GaussianShape)
    assert make_shape("bessel", 2.0, scale=0.5).rcut == 1.0
    # Halving alpha*r_c^2 broadens the Gaussian, so more of it escapes.
    assert (
        make_shape("gaussian", 1.9, scale=0.5).escaped_charge()
        > make_shape("gaussian", 1.9).escaped_charge()
    )
    with pytest.raises(ValueError, match="must be positive"):
        make_shape("bessel", 0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        make_shape("bessel", 1.9, scale=0.0)
    with pytest.raises(ValueError, match="fit inside the augmentation sphere"):
        make_shape("bessel", 1.9, scale=1.5)
    with pytest.raises(ValueError, match="unknown compensation-charge shape"):
        make_shape("sinc", 1.9)


# ------------------------------------------------------- how deep a well the shape digs
@pytest.mark.parametrize("kind", ["gaussian", "bessel"])
@pytest.mark.parametrize("scale", [1.0, 0.8])
def test_potential_at_origin_matches_direct_integration(kind, scale):
    """``v(0) = 4 pi integral g_0 r dr``, closed form against quadrature for both shapes."""
    from scipy.integrate import quad

    shape = make_shape(kind, 1.9094498728, scale=scale)
    reference = (
        4
        * np.pi
        * quad(
            lambda r: float(shape.radial(0, np.array(r))) * r,
            0.0,
            shape.containment_radius(),
            limit=400,
        )[0]
    )
    assert shape.potential_at_origin() == pytest.approx(reference, rel=1e-11)


@pytest.mark.parametrize("alpha_rc2", [6.0, 9.0, 12.0, 20.0])
def test_equivalent_width_is_the_identity_on_a_gaussian(alpha_rc2):
    """The map back onto the Gaussian scale has to be exact where the scale is defined.

    This is the calibration: the collapse threshold was measured against Gaussian widths,
    and the diagnostic is only worth anything for other shapes if it reproduces those
    widths for the Gaussian itself.
    """
    rcut = 1.9094498728
    shape = make_shape("gaussian", rcut, scale=alpha_rc2 / 9.0)
    assert equivalent_alpha_rc2(shape, rcut) == pytest.approx(alpha_rc2, rel=1e-12)


def test_the_compact_shape_at_full_radius_is_shallower_than_the_default_gaussian():
    """Which is why it is safe to make it the default: see the table in equivalent_alpha_rc2.

    The measured collapse boundary sits between an equivalent width of 8.7 and 11.4 for
    both functional forms. Silicon's bessel scales bracket it in the same place, and the
    default sits well below.
    """
    rcut = 1.9094498728
    depths = {
        scale: equivalent_alpha_rc2(make_shape("bessel", rcut, scale=scale), rcut)
        for scale in (1.0, 0.9, 0.8, 0.7)
    }
    assert depths[1.0] == pytest.approx(5.59, abs=0.02)
    assert depths[0.7] == pytest.approx(11.40, abs=0.02)
    assert depths[1.0] < equivalent_alpha_rc2(make_shape("gaussian", rcut), rcut)
    assert sorted(depths, key=depths.get, reverse=True) == [0.7, 0.8, 0.9, 1.0]
