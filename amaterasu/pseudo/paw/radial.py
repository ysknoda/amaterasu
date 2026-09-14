"""Radial quadrature and electrostatics inside an augmentation sphere.

Two things happen on the radial mesh that do not happen anywhere else in the code.

**Integration on a logarithmic mesh.**
    PAW datasets are tabulated on ``r = a(exp(d i) - 1)`` or similar, which resolves the
    nuclear region without wasting points far out. Integrating such data with Simpson's
    rule *in r* is only second-order accurate because the spacing varies; integrating in
    the uniform index ``i``, with the tabulated ``dr/di`` as the Jacobian, restores the
    fourth order. The dataset ships ``dr/di`` precisely so this can be done.

**Poisson's equation, one multipole at a time.**
    A density ``rho(r) = n_L(r) Y_L(r_hat)`` produces a potential ``v_L(r) Y_L(r_hat)``
    with

        v_L(r) = 4 pi / (2l+1) [ r^-(l+1) integral_0^r n_L(s) s^(l+2) ds
                                 + r^l integral_r^R n_L(s) s^(1-l) ds ] ,

    the standard multipole solution. This is exact for a band-limited radial function
    rather than iterative, which matters because the on-site energy is a difference of two
    large electrostatic terms and any solver noise would survive the cancellation.

    The integrand ``n_L s^(1-l)`` looks singular at the origin for ``l >= 2``, but
    ``n_L(r) ~ r^l`` there, so the product vanishes as ``r``. Values at ``r = 0`` are set
    from the limit rather than evaluated.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from ..base import RadialGrid

__all__ = [
    "cumulative_integral",
    "integrate",
    "multipole_moment",
    "radial_hartree_energy",
    "radial_poisson",
]


def _jacobian(grid: RadialGrid) -> np.ndarray | None:
    """``dr/di`` when the mesh is tabulated with one, else ``None`` for a uniform mesh."""
    if grid.dr_di is None or len(grid.dr_di) != len(grid.r):
        return None
    return np.asarray(grid.dr_di, dtype=float)


def integrate(grid: RadialGrid, values: np.ndarray, *, upto: int | None = None):
    """``integral f(r) dr`` over the mesh, or over its first ``upto`` points."""
    stop = len(grid.r) if upto is None else upto
    jacobian = _jacobian(grid)
    values = np.asarray(values, dtype=float)[..., :stop]
    if jacobian is None:
        total = simpson(values, x=grid.r[:stop])
    else:
        total = simpson(values * jacobian[:stop], dx=1.0)
    return float(total) if values.ndim == 1 else total


def _cumulative_uniform(values: np.ndarray, step: float) -> np.ndarray:
    """Running integral of samples on a uniform mesh, exact for cubics.

    Each panel is integrated by fitting a cubic through the four surrounding samples,

        integral_{x_k}^{x_k+1} f = h/24 (-f_{k-1} + 13 f_k + 13 f_k+1 - f_k+2) ,

    with the one-sided variants of the same order at the two ends. SciPy's
    ``cumulative_simpson`` is the obvious alternative but is only accurate to about
    ``1e-7`` here, and that error would survive the cancellation between the two large
    electrostatic terms of the on-site energy -- around 5 meV for silicon, which is the
    whole error budget.
    """
    y = np.asarray(values, dtype=float)
    if y.shape[-1] < 4:
        raise ValueError("cumulative integration needs at least four mesh points")
    panels = np.empty((*y.shape[:-1], y.shape[-1] - 1))
    panels[..., 0] = 9 * y[..., 0] + 19 * y[..., 1] - 5 * y[..., 2] + y[..., 3]
    panels[..., 1:-1] = -y[..., :-3] + 13 * y[..., 1:-2] + 13 * y[..., 2:-1] - y[..., 3:]
    panels[..., -1] = y[..., -4] - 5 * y[..., -3] + 19 * y[..., -2] + 9 * y[..., -1]
    panels *= step / 24.0
    out = np.zeros_like(y)
    np.cumsum(panels, axis=-1, out=out[..., 1:])
    return out


def cumulative_integral(
    grid: RadialGrid, values: np.ndarray, *, upto: int | None = None
) -> np.ndarray:
    """``integral_0^r f(s) ds`` at the first ``upto`` mesh points.

    Only the first ``upto`` samples are ever touched. That matters because the cubic panels
    span four neighbours: running them across a zero-padded tail would fit a cubic through
    an artificial step and lose half a panel of the jump, which is a far larger error than
    the scheme's own.
    """
    stop = len(grid.r) if upto is None else int(upto)
    jacobian = _jacobian(grid)
    values = np.asarray(values, dtype=float)[..., :stop]
    if jacobian is not None:
        # Uniform in the index i, with dr/di supplied by the dataset.
        return _cumulative_uniform(values * jacobian[:stop], 1.0)
    spacing = np.diff(grid.r[:stop])
    if np.allclose(spacing, spacing[0], rtol=1e-9):
        return _cumulative_uniform(values, float(spacing[0]))
    from scipy.integrate import cumulative_simpson  # pragma: no cover - unusual mesh

    return cumulative_simpson(values, x=grid.r[:stop], initial=0.0)  # pragma: no cover


def multipole_moment(grid: RadialGrid, l: int, density: np.ndarray, *, upto=None) -> float:
    """``integral n_L(r) r^(l+2) dr``, the ``l``-th multipole of ``n_L(r) Y_L``.

    ``density`` may be given either on the whole mesh or on its first ``upto`` points.
    """
    stop = len(grid.r) if upto is None else int(upto)
    values = np.asarray(density, dtype=float)[..., :stop]
    return integrate(grid, values * grid.r[:stop] ** (l + 2), upto=stop)


def radial_poisson(
    grid: RadialGrid, l: int, density: np.ndarray, *, upto: int | None = None
) -> np.ndarray:
    """Potential ``v_L(r)`` of the charge distribution ``n_L(r) Y_L(r_hat)``.

    The density is treated as zero beyond ``upto`` points, so the potential there is the
    pure multipole tail ``4 pi Q_l / ((2l+1) r^(l+1))``. That truncation is deliberate: a
    PAW on-site density is only defined inside the augmentation sphere, where the partial
    waves mean something, and several of them diverge outside it.
    """
    r = grid.r
    stop = len(r) if upto is None else int(upto)
    head = r[:stop]
    charge = np.asarray(density, dtype=float)[:stop]

    inner = cumulative_integral(grid, charge * head ** (l + 2), upto=stop)

    # integral_r^R n s^(1-l) ds, as the total minus the running integral. Written this way
    # rather than by integrating backwards so that both pieces share one quadrature.
    with np.errstate(divide="ignore", invalid="ignore"):
        outer_integrand = np.where(head > 0, charge * head ** (1 - l), 0.0)
    if l == 1:
        # r^(1-l) is 1 here, so the integrand is finite at the origin and equals the
        # density there; every other l either vanishes (l = 0) or is killed by the
        # density's own r^l behaviour (l >= 2), which np.where already returns as zero.
        outer_integrand[head == 0] = charge[head == 0]
    running = cumulative_integral(grid, outer_integrand, upto=stop)
    outer = running[-1] - running

    potential = np.zeros_like(r)
    positive = head > 0
    potential[:stop][positive] = (
        inner[positive] / head[positive] ** (l + 1) + head[positive] ** l * outer[positive]
    )
    if not positive[0]:
        # At r = 0 only the l = 0 channel is finite, and equals the outer integral there.
        potential[0] = outer[0] if l == 0 else 0.0
    # Beyond the charge there is nothing left to enclose: the pure multipole tail.
    if stop < len(r):
        potential[stop:] = inner[-1] / r[stop:] ** (l + 1)
    return (4.0 * np.pi / (2 * l + 1)) * potential


def radial_hartree_energy(
    grid: RadialGrid, densities: dict[int, np.ndarray] | np.ndarray, *, upto=None
) -> float:
    """``1/2 integral rho v_H[rho]`` for a density given by its ``Y_L`` components.

    ``densities`` maps a composite index ``L`` to ``n_L(r)``; passing a single array is
    shorthand for an ``l = 0`` (spherical) density.
    """
    if isinstance(densities, np.ndarray):
        densities = {0: densities}
    total = 0.0
    for index, component in densities.items():
        l = int(np.sqrt(index))
        potential = radial_poisson(grid, l, component, upto=upto)
        total += 0.5 * integrate(grid, component * potential * grid.r**2, upto=upto)
    return float(total)


def derivative(grid: RadialGrid, values: np.ndarray) -> np.ndarray:
    """``df/dr`` on the mesh, by a fourth-order difference in the uniform index.

    Radial data is smooth in the index, not in ``r``, so the derivative is taken there and
    divided by the tabulated ``dr/di``. A five-point central stencil is used inside and
    one-sided stencils of the same order at the two ends.
    """
    y = np.asarray(values, dtype=float)
    n = y.shape[-1]
    if n < 5:
        raise ValueError("the fourth-order derivative needs at least five mesh points")
    d = np.empty_like(y)
    d[..., 2:-2] = (y[..., :-4] - 8 * y[..., 1:-3] + 8 * y[..., 3:-1] - y[..., 4:]) / 12.0
    forward = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / 12.0
    for k in (0, 1):
        d[..., k] = np.einsum("j,...j->...", forward, y[..., k : k + 5])
    for k in (n - 2, n - 1):
        d[..., k] = -np.einsum("j,...j->...", forward, y[..., k : k - 5 : -1])
    jacobian = _jacobian(grid)
    if jacobian is None:
        spacing = np.diff(grid.r)
        if not np.allclose(spacing, spacing[0], rtol=1e-9):  # pragma: no cover
            raise ValueError("derivative needs a uniform mesh or a tabulated dr/di")
        jacobian = np.full(len(grid.r), spacing[0])
    if n > len(jacobian):
        raise ValueError(f"values have {n} points but the mesh has {len(jacobian)}")
    return d / jacobian[:n]


def poisson_operator(grid: RadialGrid, l: int, *, upto: int, measure: np.ndarray) -> np.ndarray:
    """A dense radial Poisson operator whose quadratic form is exactly symmetric.

    :func:`radial_poisson` is accurate but its *discrete* operator is not self-adjoint:
    the running integral over the source uses the four-point cumulative weights while the
    outer integral over the field point uses Simpson's, and those two weight vectors
    differ pointwise by up to a factor of two even though both integrate smooth functions
    to the same order. The electrostatic energy is unaffected -- it is a scalar and the
    two quadratures agree on it -- but its *gradient* is not: ``dE/dn`` is
    ``(W P + P^T W) n / 2``, and dropping the second half leaves a several-per-cent error
    in the on-site Hamiltonian. That error is invisible to an energy check and shows up
    only as a self-consistent solution that is not the stationary point of the energy
    being reported.

    The fix costs nothing in accuracy. With ``W`` the volume measure ``w_r r^2``, the
    symmetrised operator is ``W^-1 (W P + P^T W)/2``, and since ``n^T P^T W n`` and
    ``n^T W P n`` are the same scalar, the energy it gives is *identical* to
    :func:`radial_poisson`'s -- bit for bit -- while the gradient becomes exact.

    ``measure`` is ``W``; its first entry is zero when the mesh starts at ``r = 0``, and
    that row of the operator is set to zero rather than divided by it, which is harmless
    because every use of the potential is weighted by ``W``.

    One caveat for anyone reusing this: ``operator @ n`` is *not* the physical potential of
    ``n``. Symmetrising changes the map even though it preserves the quadratic form, and on
    a test charge the two differ by tens of per cent. This object exists to supply energies
    and their gradients, both of which only ever see the symmetric part; for a potential to
    look at or to plot, call :func:`radial_poisson`.
    """
    size = int(upto)
    measure = np.asarray(measure, dtype=float)[:size]
    columns = np.empty((size, size))
    unit = np.zeros(size)
    for k in range(size):
        unit[k] = 1.0
        columns[:, k] = radial_poisson(grid, l, unit, upto=size)[:size]
        unit[k] = 0.0
    symmetric = 0.5 * (measure[:, None] * columns + columns.T * measure[None, :])
    safe = np.where(measure > 0, measure, 1.0)
    operator = symmetric / safe[:, None]
    operator[measure <= 0] = 0.0
    return operator
