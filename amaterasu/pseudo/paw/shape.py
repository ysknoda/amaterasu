"""Compensation-charge shape functions.

PAW replaces the true charge inside an augmentation sphere with a smooth one carrying the
same electrostatic multipole moments. The moments are fixed by the dataset and the
occupancies; the *shape* that carries them is a choice made by the code, not by the
dataset, and in exact arithmetic the total energy does not depend on it -- the same
function appears in the smooth plane-wave term and is subtracted again in the one-centre
term, so it cancels. What it does control is how many plane waves the compensation charge
needs, and how exactly that cancellation survives being carried out in two different
bases.

The normalisation is Blochl's, and is the one the multipole bookkeeping assumes:

    integral d^3r r^l g_L(r) Y_L(r_hat) = 1   <=>   integral_0^inf g_l(r) r^(l+2) dr = 1 ,

so that a compensation charge ``sum_L Q_L g_l(r) Y_L`` has exactly the moments ``Q_L``.

Two shapes are implemented. :class:`GaussianShape` is the default, and
:class:`BesselShape` is compactly supported. Neither is right; the rest of this docstring is
the measured difference between them, because the choice is not obvious and the numbers are
the only honest way to describe it.

The Gaussian, and the two bounds on its width
---------------------------------------------
A Gaussian's Fourier transform is closed-form and decays *exponentially*, so the
compensation charge is resolved on the plane-wave grid to machine precision at any sane
cutoff. Against that it has no compact support, and it is bounded from both sides.

*From above*, because the cancellation between the smooth and one-centre terms holds only
while the shape is fully accounted for on both sides. Truncating the one-centre radial
integrals at ``r_c`` instead of at :meth:`GaussianShape.containment_radius` costs about
95 meV/atom for silicon at the default width -- thirty times the delta-factor gate.
Integrating out to the containment radius fixes that, at the price of carrying the partial
waves further out than they mean anything (see :mod:`amaterasu.pseudo.paw.dataset`).

*From below*, and much more sharply, because the compensation charge is *negative* and
carries the nucleus, so concentrating it digs an attractive well for the smooth orbitals;
they fall in, which raises ``D_ij``, which makes ``Q_00`` more negative, which deepens the
well. The term meant to cancel it, ``sum_L Delta_Lij W^a_L``, is only linear in ``D_ij`` and
cannot keep up. Measured for silicon at ecut = 20 Ha, k = 4^3:

    alpha r_c^2 |  ecut_rho = 80 Ha        |  ecut_rho = 160 Ha
    ------------+--------------------------+--------------------------
        6       |  -15657.703, 16 it       |  -15657.712, 16 it
        9       |  -15657.761, 17 it       |  -15657.797, 48 it
       12       |  -15657.788, 19 it       |  collapsed
       20       |  collapsed               |  collapsed

Note which way the cutoff pushes: ``alpha r_c^2 = 12`` is fine on the coarser density grid
and collapses on the finer one. A larger basis resolves the well *better*, so it falls
further -- the mark of a functional that is unbounded below rather than of a shape the grid
cannot represent. That leaves a usable window of roughly 6 to 9, and the total energy moves
by about 85 meV/atom across it, when it should not move at all.

The Bessel shape
----------------
:class:`BesselShape` is exactly zero outside its own radius, so nothing escapes, nothing
lands inside a neighbour's augmentation sphere where the one-centre expansion says nothing,
and the one-centre integrals are complete without being extended anywhere. It is the
``bessel`` form the JTH datasets themselves declare, and follows Blochl, Phys. Rev. B 50,
17953 (1994), section IV: two spherical Bessel functions whose arguments are chosen so that
each one vanishes at ``r_c``, combined so that the sum has a vanishing derivative there too.

Its form factor decays *algebraically*, as ``q^-5`` with oscillations. That sounds like it
should cost a much larger density cutoff and does not: the fraction of the compensation
charge's Hartree self-energy left outside ``ecut_rho = 160 Ha`` is 4.3e-11, against the
Gaussian's 1.9e-30, and both are irrelevant. Measured rather than argued -- aluminium's
total energy moves by 0.008 meV/atom between 160 and 320 Ha.

Compact support removes the leak; it does not remove the *lower* bound. Filling the whole
sphere the Bessel shape is shallower than the default Gaussian, but shrink its support to
``0.7 r_c`` and silicon collapses exactly as a Gaussian at ``alpha r_c^2 = 12`` does. That
is one phenomenon, not two: :func:`equivalent_alpha_rc2` measures the depth of the well any
shape digs on the one scale the table above calibrates, and puts both collapses between 8.7
and 11.4. :func:`amaterasu.hamiltonian.paw.build_paw_machinery` warns before a run crosses
it.

What is left, and where it comes from
-------------------------------------
Removing the leak does not remove the shape dependence. Silicon at ecut = 20 Ha, k = 4^3,
with the Bessel support scaled from ``r_c`` down to ``0.8 r_c``:

    support     ecut_rho = 160 Ha       ecut_rho = 320 Ha
    --------    ---------------------   ---------------------
    1.00 r_c    -15657.748051, 19 it    -15657.747853, 18 it
    0.90 r_c    -15657.756728, 18 it    -15657.756538, 18 it
    0.80 r_c    -15657.773900, 21 it    -15657.773712, 23 it
    spread          25.85 meV               25.86 meV

25.9 meV/atom where there should be none -- about a third of the Gaussian's 85 over a
comparable fractional change of width. The two columns are the important part: doubling the
density cutoff moves the *absolute* energy by 0.19 meV/atom and the spread by 0.01, so this
is not the plane-wave basis failing to resolve the shape and no cutoff will remove it.

The reason is visible in the energy expression. Once the shape is compact, ``Z~^a`` survives
in exactly one place: the difference between the plane-wave ``integral n~ v[Z~^a]`` and the
one-centre ``integral n~^a v[Z~^a]``. Outside the support those agree, because ``v[Z~^a]``
there is the pure multipole field ``Q_L/r^(l+1)`` and has forgotten the shape entirely --
that is what compact support buys, and what a Gaussian's tail spoils. Inside, they differ by
``n~ - n~^a``, the amount by which the dataset's partial waves fail to span the smooth
density there, and that difference weights a potential which *is* shape-dependent. The JTH
silicon set carries two ``s`` and two ``p`` channels and no ``d``, so inside the sphere of a
tetrahedrally bonded atom there is real density it cannot represent.

That explanation is testable by changing the dataset rather than the shape, and it holds.
Silicon again, same code and same settings, with the pslibrary ``kjpaw`` table, which
carries the same ``s`` and ``p`` channels *plus* two ``d``:

    silicon dataset          partial waves      B0 error    delta factor
    JTH PAW-XML              s s p p             -6.1%       1.868 meV/atom
    pslibrary kjpaw          s s p p d d         +0.9%       0.326 meV/atom

Both rows use the default Gaussian, so the shape is held fixed and only the dataset moves.
The bulk modulus error and most of the delta factor go away when the ``d`` channel is
added, and neither has anything to do with the compensation charge. The residual is a
property of the dataset, not of this module.

What that does *not* license is the further step of predicting which shape wins from the
partial waves alone; see the next section, where the pslibrary silicon row refutes it.

Why the Gaussian is still the default
-------------------------------------
Because the two shapes trade places depending on the element, no rule was found that
predicts which one wins, and the Gaussian is the one that clears every gate everywhere it
was tried. Six element/dataset combinations, all at identical cutoffs, k-points and
thresholds, ordered by atomic number:

    element    dataset      partial waves   gaussian   bessel    delta factor, meV/atom
    O   (8)    JTH          s s p p           0.871     2.462    gate < 3
    Al (13)    JTH          s s p p           0.200     2.048
    Si (14)    JTH          s s p p           1.868     2.378
    Si (14)    pslibrary    s s p p d d       0.326     0.658
    Fe (26)    JTH          s s p p d d       2.699     0.825
    Cu (29)    JTH          s s p p d d       0.391     0.064

Both shapes clear the gate in all six. The Gaussian wins four of them; the Bessel shape
wins the two 3d transition metals, by 3.3x for iron and 6x for copper.

The obvious structural reading of the section above -- that the compact shape should win
exactly when the dataset carries ``d`` partial waves, since those are what shrink the
``n~ - n~^a`` residual the shape multiplies -- was the working hypothesis, and it is
**wrong**. The pslibrary silicon row refutes it: that table carries two ``d`` channels and
the Gaussian still wins by a factor of two. Partial-wave completeness explains why the
*absolute* residual is smaller for some datasets than others, but not which shape comes out
ahead.

What the six rows actually split on is the element -- main group against 3d transition
metal -- and pslibrary silicon is exactly the row that separates that from the
partial-wave reading, since it has the ``d`` channels and still sides with the main group.
Six points is a description, not a rule, and no mechanism for it was identified here.

The secondary gate points the same way. The lattice constant has to agree within 0.01 A
with a norm-conserving calculation of the same element, and the single failure anywhere is
the Bessel shape on the JTH silicon table:

    silicon dataset      gaussian            bessel              (norm-conserving 5.4700 A)
    JTH PAW-XML          5.4783 A  +0.0083   5.4805 A  +0.0105   <- the one failure
    pslibrary kjpaw      5.4709 A  +0.0009   5.4724 A  +0.0024

which makes that failure a property of one dataset rather than of the shape -- and it still
happens only to the shape that is not the default.

Against that the Bessel shape keeps two advantages the new elements did not touch. It is
much better on the *curvature* of the equation of state, which is the part of E(V) that
reflects the energy functional rather than an offset (silicon's bulk modulus error 6.1% ->
0.7%), and better against the dataset's own all-electron eigenvalues, the one number here
that does not come from amaterasu (isolated silicon 3p-3s error 117.7 -> 68.6 meV, reached
in 28 SCF iterations instead of 52).

One cost falls only on the compact shape, and it is worth stating because it is a
property of the form factor rather than of any element. The Bessel envelope decays
algebraically, so the density cutoff needed to resolve the compensation charge scales with
``1/r_c`` -- and oxygen has the smallest ``r_c`` of the six. What
:func:`amaterasu.hamiltonian.paw._unresolved_multipole` reports, which is the worst
multipole's envelope over the last oscillation below the cutoff:

    ecut_rho      160 Ha                   320 Ha
                O        Si       Cu      O        Si       Cu
    gaussian  1.9e-05  4.4e-11  2.4e-12  4.9e-12  9.5e-24  2.4e-26
    bessel    2.5e-04  2.3e-05  2.0e-05  4.0e-05  4.1e-06  2.7e-06

Oxygen with the Bessel shape is the only case measured here that crosses the 1e-4 the
runtime warning fires at, and doubling the density cutoff to 320 Ha clears it. The
Gaussian's exponential decay makes 160 Ha ample for every element.

Mixed evidence is not a reason to move a default that meets its gates. The Gaussian stays;
the Bessel shape is a verified alternative, selected with ``pseudo.compensation_shape:
bessel``, and it is the one to reach for on a 3d transition metal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, gammainc, spherical_jn

__all__ = [
    "BesselShape",
    "GaussianShape",
    "ShapeFunction",
    "double_factorial",
    "equivalent_alpha_rc2",
    "make_shape",
]

#: alpha * r_c^2 for the default Gaussian width; see the module docstring for why it is
#: bounded from *both* sides.
DEFAULT_ALPHA_RC2 = 9.0

#: Half-width of the window, in units of ``|x - z_i|``, inside which :class:`BesselShape`
#: switches from dividing by ``z_i^2 - x^2`` to a Taylor series about the zero; see
#: :func:`_removable`. It sits where the two branches are equally wrong: the quotient's
#: error grows as ``1e-16/|x-z|`` going in, the series' as ``|x-z|^4`` going out. Scanned
#: against adaptive quadrature across the switch for l = 0..4, the worst relative error
#: anywhere -- including exactly on a zero -- is 4.6e-13, and it is reached at the boundary
#: itself.
_PATCH_WINDOW = 2.0e-3


def double_factorial(l: int) -> float:
    """``(2l+1)!! = 1, 3, 15, 105, ...``"""
    result = 1.0
    for k in range(1, 2 * l + 2, 2):
        result *= k
    return result


@cache
def spherical_jn_zeros(l: int, count: int) -> tuple[float, ...]:
    """The first ``count`` positive zeros of ``j_l``.

    Bracketed by the interlacing property -- between two consecutive zeros of ``j_(l-1)``
    there is exactly one zero of ``j_l`` -- starting from the zeros of ``j_0``, which are
    the multiples of pi. That makes every bracket provably correct, so the root finder
    cannot land on the wrong zero, which a blind scan of ``j_l`` could.
    """
    zeros = np.pi * np.arange(1, count + l + 1)
    for order in range(1, l + 1):
        bracket = zeros

        def value(x: float, l: int = order) -> float:
            return float(spherical_jn(l, x))

        zeros = np.array(
            [brentq(value, bracket[n], bracket[n + 1], xtol=1e-15) for n in range(len(bracket) - 1)]
        )
    return tuple(float(z) for z in zeros[:count])


def _reduced(l: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``f(x) = j_l(x)/x^l`` and ``f'(x)/x``, both finite at the origin.

    ``f`` is even and entire, which is what makes every expression below well conditioned
    at ``q = 0``; dividing ``j_l`` by ``x^l`` numerically is not, so the origin is filled in
    from the series limits ``f(0) = 1/(2l+1)!!`` and ``f'(0)/0 = -1/(2l+3)!!``.
    """
    safe = np.where(x > 0, x, 1.0)
    value = np.where(x > 0, spherical_jn(l, safe) / safe**l, 1.0 / double_factorial(l))
    # f'(x) = -j_(l+1)(x)/x^l follows from d/dx [x^-l j_l] = -x^-l j_(l+1).
    slope_over_x = np.where(
        x > 0,
        -spherical_jn(l + 1, safe) / safe ** (l + 1),
        -1.0 / double_factorial(l + 1),
    )
    return value, slope_over_x


def _removable(l: int, z: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``S(x) = f(x)/(z^2 - x^2)`` and ``S'(x)/x``, where ``z`` is a zero of ``j_l``.

    ``f`` vanishes at ``z``, so ``S`` is analytic there, but the direct quotient is not:
    ``f`` is computed to about 1e-16 *absolute*, so its relative accuracy near its own zero
    degrades as ``1e-16/|x-z|``, and at ``x = z`` exactly it is 0/0. Inside
    :data:`_PATCH_WINDOW` the quotient is replaced by the series about ``z``,

        S(x) = -[f'(z) + f''(z) e/2 + f'''(z) e^2/6 + f''''(z) e^3/24] / (2z + e) ,

    with ``e = x - z``, which divides only by ``2z + e`` and is therefore exact to the
    truncation error. The derivatives come from ``f_m' = -x f_(m+1)`` with
    ``f_m(x) = j_m(x)/x^m``, applied four times.
    """
    x = np.asarray(x, dtype=float)
    value = np.empty_like(x)
    slope_over_x = np.empty_like(x)

    near = np.abs(x - z) < _PATCH_WINDOW
    far = ~near

    if far.any():
        f, fp_over_x = _reduced(l, x[far])
        u = z**2 - x[far] ** 2
        value[far] = f / u
        slope_over_x[far] = fp_over_x / u + 2.0 * f / u**2

    if near.any():
        m = np.array([z])
        f1 = float(spherical_jn(l + 1, m)[0]) / z ** (l + 1)
        f2 = float(spherical_jn(l + 2, m)[0]) / z ** (l + 2)
        f3 = float(spherical_jn(l + 3, m)[0]) / z ** (l + 3)
        f4 = float(spherical_jn(l + 4, m)[0]) / z ** (l + 4)
        d1 = -z * f1
        d2 = -f1 + z**2 * f2
        d3 = 3.0 * z * f2 - z**3 * f3
        d4 = 3.0 * f2 - 6.0 * z**2 * f3 + z**4 * f4

        e = x[near] - z
        series = d1 + d2 * e / 2.0 + d3 * e**2 / 6.0 + d4 * e**3 / 24.0
        derivative = d2 / 2.0 + d3 * e / 3.0 + d4 * e**2 / 8.0
        denominator = 2.0 * z + e
        value[near] = -series / denominator
        slope_over_x[near] = (-derivative / denominator + series / denominator**2) / x[near]

    return value, slope_over_x


@cache
def _bessel_terms(l: int, rcut: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """``(a_1, a_2)`` and ``(z_1, z_2)`` for ``g_l = sum_i a_i j_l(z_i r / r_c)``.

    ``z_i`` are the first two zeros of ``j_l``, so each term vanishes at ``r_c`` on its own.
    The amplitudes are then fixed by the two remaining conditions: the sum has a vanishing
    derivative at ``r_c`` (so the shape meets zero smoothly rather than with a kink), and
    carries unit multipole. Both are linear in ``a``, and with

        integral_0^rc j_l(q_i r) r^(l+2) dr = r_c^(l+2) j_(l+1)(z_i) / q_i ,
        d/dr j_l(q_i r)|_rc = -q_i j_(l+1)(z_i) ,

    -- the second because ``j_l'(z) = -j_(l+1)(z)`` at a zero of ``j_l`` -- both reduce to
    conditions on ``b_i = a_i j_(l+1)(z_i)`` alone, which solve in closed form.
    """
    z = spherical_jn_zeros(l, 2)
    q = (z[0] / rcut, z[1] / rcut)
    b1 = q[0] * q[1] ** 2 / (rcut ** (l + 2) * (q[1] ** 2 - q[0] ** 2))
    b2 = -b1 * q[0] / q[1]
    tail = spherical_jn(l + 1, np.array(z))
    return (float(b1 / tail[0]), float(b2 / tail[1])), z


class ShapeFunction(Protocol):
    """What the rest of PAW needs from a compensation-charge shape.

    ``radial`` is the one-centre side and ``form_factor`` the plane-wave side of the same
    function; they have to describe the *same* charge to the accuracy the total energy is
    quoted at, since the cancellation between the two terms is what PAW rests on.
    ``envelope`` and ``envelope_slope_over_q`` exist so the stress tensor can differentiate
    the plane-wave side without ever forming ``q^l`` divided by itself.
    """

    #: Radius the shape was built for, in Bohr.
    rcut: float

    def radial(self, l: int, r: np.ndarray) -> np.ndarray: ...
    def form_factor(self, l: int, q: np.ndarray) -> np.ndarray: ...
    def form_factor_derivative(self, l: int, q: np.ndarray) -> np.ndarray: ...
    def envelope(self, l: int, q: np.ndarray) -> np.ndarray: ...
    def envelope_slope_over_q(self, l: int, q: np.ndarray) -> np.ndarray: ...
    def escaped_charge(self, rcut: float | None = None) -> float: ...
    def containment_radius(self, tolerance: float = 1e-12) -> float: ...
    def enclosed_monopole(self, r: np.ndarray) -> np.ndarray: ...
    def potential_at_origin(self) -> float: ...


@dataclass(frozen=True)
class BesselShape:
    """``g_l(r) = sum_i a_i j_l(z_i r/r_c)`` for ``r <= r_c`` and exactly zero beyond.

    The compact support is the point: nothing of the compensation charge lies outside the
    augmentation sphere, so the one-centre term subtracts back exactly what the smooth term
    added, and the total energy stops depending on a function it should not depend on. See
    the module docstring for what that buys and what it costs.
    """

    #: Support radius, beyond which the shape is identically zero.
    rcut: float

    def __str__(self) -> str:
        return (
            f"Bessel compensation charge, compact inside rc={self.rcut:.3f} Bohr "
            f"(zeros of j_0 at {spherical_jn_zeros(0, 2)[0]:.4f}, "
            f"{spherical_jn_zeros(0, 2)[1]:.4f})"
        )

    def _wavenumbers(self, l: int) -> tuple[float, float]:
        return tuple(z / self.rcut for z in spherical_jn_zeros(l, 2))  # type: ignore[return-value]

    def radial(self, l: int, r: np.ndarray) -> np.ndarray:
        """``g_l(r)`` on a radial mesh, zero outside ``r_c``."""
        amplitudes, _ = _bessel_terms(l, self.rcut)
        r = np.asarray(r, dtype=float)
        inside = np.minimum(r, self.rcut)  # keep j_l's argument bounded, then mask
        q = self._wavenumbers(l)
        value = amplitudes[0] * spherical_jn(l, q[0] * inside)
        value += amplitudes[1] * spherical_jn(l, q[1] * inside)
        return np.where(r <= self.rcut, value, 0.0)

    def _kernel(self, l: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``P(x) = f(x)/((z_1^2-x^2)(z_2^2-x^2))`` and ``P'(x)/x``, with ``f = j_l/x^l``.

        Only one of the two factors can be small at any ``x``, so the removable singularity
        is taken out of whichever zero is nearer and the other factor is divided by
        directly. The switch is at the midpoint of the two zeros, where both branches are
        equally well conditioned.
        """
        z = spherical_jn_zeros(l, 2)
        x = np.asarray(x, dtype=float)
        value = np.empty_like(x)
        slope_over_x = np.empty_like(x)
        lower = x <= 0.5 * (z[0] + z[1])

        for mask, near, other in ((lower, z[0], z[1]), (~lower, z[1], z[0])):
            if not mask.any():
                continue
            reduced, reduced_slope = _removable(l, near, x[mask])
            rest = other**2 - x[mask] ** 2
            value[mask] = reduced / rest
            slope_over_x[mask] = reduced_slope / rest + 2.0 * reduced / rest**2
        return value, slope_over_x

    def envelope(self, l: int, q: np.ndarray) -> np.ndarray:
        """``h(q) = g~_l(q)/q^l``, finite and even at the origin.

        In closed form the Lommel integral of two spherical Bessel functions collapses,
        because ``j_l(z_i) = 0`` kills one of its two terms, to

            g~_l(q) = q^l z_1^2 z_2^2 f(x) / [(z_1^2 - x^2)(z_2^2 - x^2)] ,  x = q r_c ,

        with ``f(x) = j_l(x)/x^l``. At ``q = 0`` this is ``1`` for ``l = 0`` and ``0``
        otherwise, so the monopole lands exactly on the ``G = 0`` coefficient and the cell
        stays neutral to machine precision, exactly as for the Gaussian.
        """
        z = spherical_jn_zeros(l, 2)
        value, _ = self._kernel(l, np.asarray(q, dtype=float) * self.rcut)
        return z[0] ** 2 * z[1] ** 2 * value

    def envelope_slope_over_q(self, l: int, q: np.ndarray) -> np.ndarray:
        """``h'(q)/q``, finite at the origin because ``h`` is even in ``q``."""
        z = spherical_jn_zeros(l, 2)
        _, slope = self._kernel(l, np.asarray(q, dtype=float) * self.rcut)
        return z[0] ** 2 * z[1] ** 2 * self.rcut**2 * slope

    def form_factor(self, l: int, q: np.ndarray) -> np.ndarray:
        """``g~_l(q) = integral g_l(r) j_l(q r) r^2 dr``."""
        q = np.asarray(q, dtype=float)
        return q**l * self.envelope(l, q)

    def form_factor_derivative(self, l: int, q: np.ndarray) -> np.ndarray:
        """``d g~_l / dq``, assembled from the envelope so nothing divides by ``q``."""
        q = np.asarray(q, dtype=float)
        radial = l * q ** (l - 1) if l > 0 else np.zeros_like(q)
        return radial * self.envelope(l, q) + q ** (l + 1) * self.envelope_slope_over_q(l, q)

    def enclosed_monopole(self, r: np.ndarray) -> np.ndarray:
        """``integral_0^r g_0 s^2 ds``, from ``integral_0^r j_0(q s) s^2 ds = r^2 j_1(q r)/q``."""
        amplitudes, _ = _bessel_terms(0, self.rcut)
        r = np.asarray(r, dtype=float)
        inside = np.minimum(r, self.rcut)
        q = self._wavenumbers(0)
        value = amplitudes[0] * inside**2 * spherical_jn(1, q[0] * inside) / q[0]
        value += amplitudes[1] * inside**2 * spherical_jn(1, q[1] * inside) / q[1]
        return np.where(r >= self.rcut, 1.0, value)

    def escaped_charge(self, rcut: float | None = None) -> float:
        """Fraction of the ``l = 0`` charge outside ``rcut``; exactly zero at the support."""
        radius = self.rcut if rcut is None else rcut
        if radius >= self.rcut:
            return 0.0
        return float(1.0 - self.enclosed_monopole(np.array(radius)))

    def containment_radius(self, tolerance: float = 1e-12) -> float:
        """The support radius itself: there is nothing beyond it to contain."""
        return self.rcut

    def potential_at_origin(self) -> float:
        """``4 pi integral g_0(r) r dr``: how deep a well this shape digs. See
        :func:`equivalent_alpha_rc2`.

        Closed form from ``integral_0^rs j_0(q r) r dr = (1 - cos(q rs))/q^2``. For ``l = 0``
        the two ``q_i r_s`` are pi and 2 pi, so the second term contributes nothing.
        """
        amplitudes, zeros = _bessel_terms(0, self.rcut)
        q = self._wavenumbers(0)
        return float(
            4.0
            * np.pi
            * sum(
                a * (1.0 - np.cos(z)) / qi**2 for a, z, qi in zip(amplitudes, zeros, q, strict=True)
            )
        )


@dataclass(frozen=True)
class GaussianShape:
    """``g_l(r) = N_l r^l exp(-alpha r^2)`` with the Blochl multipole normalisation.

    Kept as an option, and as the reference the module docstring's measurements were made
    against, but not the default: it has no compact support, and the module docstring
    explains why no width setting repairs that.
    """

    alpha: float
    #: The augmentation radius the width was derived from, for diagnostics.
    rcut: float = 0.0

    @classmethod
    def for_radius(cls, rcut: float, *, alpha_rc2: float = DEFAULT_ALPHA_RC2) -> GaussianShape:
        if rcut <= 0:
            raise ValueError(f"the augmentation radius must be positive, got {rcut}")
        return cls(alpha=alpha_rc2 / rcut**2, rcut=float(rcut))

    def __str__(self) -> str:
        return (
            f"Gaussian compensation charge, alpha={self.alpha:.4f} Bohr^-2 "
            f"(rc={self.rcut:.3f} Bohr, {self.escaped_charge():.2e} of the monopole outside)"
        )

    def normalisation(self, l: int) -> float:
        """``N_l``, fixed by ``integral g_l r^(l+2) dr = 1``."""
        from scipy.special import gamma

        return 2.0 * self.alpha ** (l + 1.5) / gamma(l + 1.5)

    def radial(self, l: int, r: np.ndarray) -> np.ndarray:
        """``g_l(r)`` on a radial mesh."""
        r = np.asarray(r, dtype=float)
        return self.normalisation(l) * r**l * np.exp(-self.alpha * r**2)

    def form_factor(self, l: int, q: np.ndarray) -> np.ndarray:
        """``g~_l(q) = integral g_l(r) j_l(q r) r^2 dr``, in closed form.

        At ``q = 0`` this is ``1`` for ``l = 0`` and ``0`` otherwise, so the monopole of
        the compensation charge lands exactly on the ``G = 0`` coefficient and the cell
        stays neutral to machine precision.
        """
        q = np.asarray(q, dtype=float)
        return q**l * np.exp(-(q**2) / (4.0 * self.alpha)) / double_factorial(l)

    def form_factor_derivative(self, l: int, q: np.ndarray) -> np.ndarray:
        """``d g~_l / dq``, which the stress tensor needs."""
        q = np.asarray(q, dtype=float)
        envelope = np.exp(-(q**2) / (4.0 * self.alpha)) / double_factorial(l)
        radial = l * q ** (l - 1) if l > 0 else np.zeros_like(q)
        return envelope * (radial - q ** (l + 1) / (2.0 * self.alpha))

    def envelope(self, l: int, q: np.ndarray) -> np.ndarray:
        """``h(q) = g~_l(q) / q^l``, the part of the form factor that is not angular.

        Finite at ``q = 0``, unlike ``g~_l`` divided by ``q^l`` computed numerically, which
        is what makes the strain derivative of the compensation charge well conditioned:
        the angular dependence is carried by the solid harmonic ``C_lm(q) = q^l Y_lm``,
        which is a polynomial, and everything singular cancels between the two.
        """
        q = np.asarray(q, dtype=float)
        return np.exp(-(q**2) / (4.0 * self.alpha)) / double_factorial(l)

    def envelope_slope_over_q(self, l: int, q: np.ndarray) -> np.ndarray:
        """``h'(q)/q = -h(q)/(2 alpha)``, finite at the origin by construction."""
        return -self.envelope(l, q) / (2.0 * self.alpha)

    def escaped_charge(self, rcut: float | None = None) -> float:
        """Fraction of the ``l = 0`` charge lying outside ``rcut``.

        This is the one number that says whether the width is safe: the one-centre terms
        are only evaluated inside the sphere, so whatever spills out is represented on the
        plane-wave grid but not subtracted again, and the cancellation is that much less
        exact.
        """
        radius = self.rcut if rcut is None else rcut
        if radius <= 0:
            return float("nan")
        # int_0^x t^2 exp(-t^2) dt / int_0^inf = P(3/2, x^2), the regularised lower
        # incomplete gamma; the escaped fraction is its complement.
        return float(1.0 - gammainc(1.5, self.alpha * radius**2))

    def containment_radius(self, tolerance: float = 1e-12) -> float:
        """Radius enclosing all but ``tolerance`` of the ``l = 0`` charge.

        The one-centre integrals must reach this far, not merely to ``r_c``, or the part of
        the compensation charge beyond the sphere is added by the smooth term and never
        subtracted again. Solved from ``escaped_charge`` by bisection rather than in closed
        form: the inverse incomplete gamma is not worth a dependency for a number that is
        computed once per species.
        """
        low, high = self.rcut, max(self.rcut, 1.0)
        while self.escaped_charge(high) > tolerance:
            high *= 1.5
            if high > 1e4 * max(self.rcut, 1.0):  # pragma: no cover - defensive
                raise ValueError("the shape function does not decay")
        for _ in range(200):
            middle = 0.5 * (low + high)
            if self.escaped_charge(middle) > tolerance:
                low = middle
            else:
                high = middle
        return float(high)

    def enclosed_monopole(self, r: np.ndarray) -> np.ndarray:
        """``integral_0^r g_0 s^2 ds``: the charge inside ``r``, in closed form."""
        x = np.sqrt(self.alpha) * np.asarray(r, dtype=float)
        return erf(x) - 2.0 / np.sqrt(np.pi) * x * np.exp(-(x**2))

    def potential_at_origin(self) -> float:
        """``4 pi integral g_0 r dr = 8 sqrt(pi alpha)``. See :func:`equivalent_alpha_rc2`."""
        return float(8.0 * np.sqrt(np.pi * self.alpha))


def make_shape(kind: str, rcut: float, *, scale: float = 1.0) -> ShapeFunction:
    """Build the compensation-charge shape named by ``kind`` for an augmentation radius.

    ``scale`` is one dimensionless knob with the same meaning for both shapes: 1 is the
    default shape, and moving it away from 1 deforms that shape without changing anything
    physical. For ``bessel`` it multiplies the support radius, which must stay inside the
    augmentation sphere; for ``gaussian`` it multiplies ``alpha r_c^2``, so *smaller*
    values are broader. It exists to be varied: the total energy should not depend on it,
    and the physics suite measures how much it still does.

    It is bounded on the concentrated side for both shapes, and by the same mechanism --
    for silicon, a bessel scale of 0.7 collapses just as a Gaussian ``alpha r_c^2`` of 12
    does. :func:`equivalent_alpha_rc2` is the number that predicts it, and
    :func:`amaterasu.hamiltonian.paw.build_paw_machinery` warns when a run crosses it.
    """
    if rcut <= 0:
        raise ValueError(f"the augmentation radius must be positive, got {rcut}")
    if scale <= 0:
        raise ValueError(f"the compensation-charge scale must be positive, got {scale}")
    if kind == "bessel":
        if scale > 1.0:
            raise ValueError(
                f"a bessel compensation charge must fit inside the augmentation sphere, so "
                f"its scale cannot exceed 1; got {scale}"
            )
        return BesselShape(rcut=float(rcut) * float(scale))
    if kind == "gaussian":
        return GaussianShape.for_radius(rcut, alpha_rc2=DEFAULT_ALPHA_RC2 * float(scale))
    raise ValueError(f"unknown compensation-charge shape {kind!r}; expected 'bessel' or 'gaussian'")


def equivalent_alpha_rc2(shape: ShapeFunction, rcut: float) -> float:
    """The ``alpha r_c^2`` of the Gaussian that digs an equally deep well.

    The variational collapse described in the module docstring is not a property of the
    Gaussian: it is what happens when the compensation charge -- which is negative, and
    carries the nucleus -- is concentrated enough that the smooth orbitals fall into its
    own potential, and the only term that can push back is linear in ``D_ij``. Any shape
    can do it. What decides is how deep the well is, and
    ``v(0) = 4 pi integral g_0 r dr`` measures that directly for any shape.

    Mapping ``v(0)`` back onto the Gaussian's width, through ``v(0) = 8 sqrt(pi alpha)``,
    puts every shape on the one scale that has been measured. For silicon at r_c = 1.909
    Bohr, with the equivalent width in the last column:

        shape                 v(0)     equivalent alpha r_c^2   outcome
        gaussian alpha rc2=6  18.19            6.0              converges
        bessel   scale 1.00   17.55            5.6              converges
        bessel   scale 0.90   19.50            6.9              converges
        gaussian alpha rc2=9  22.28            9.0              converges
        bessel   scale 0.80   21.94            8.7              converges
        bessel   scale 0.70   25.07           11.4              COLLAPSES
        gaussian alpha rc2=12 25.72           12.0              COLLAPSES
        gaussian alpha rc2=20 33.21           20.0              COLLAPSES

    The boundary falls between 8.7 and 11.4 for *both* functional forms, which is the
    evidence that this single number is what matters. :data:`COLLAPSE_ALPHA_RC2` is set
    inside it. Note where the default sits: the compact shape at full radius is *shallower*
    than the Gaussian everyone uses, which is why it converges in fewer iterations.
    """
    return float((shape.potential_at_origin() * rcut / (8.0 * np.sqrt(np.pi))) ** 2)


#: Equivalent width above which the energy functional is at risk of being unbounded below;
#: see :func:`equivalent_alpha_rc2` for the measurements this sits between.
COLLAPSE_ALPHA_RC2 = 10.0
