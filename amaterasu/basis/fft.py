"""FFT grid, sphere-to-grid mapping, and the transforms between them.

Normalisation
-------------
A periodic function is represented as ``u(r) = sum_G c_G exp(i G.r)`` with no volume
factor, so the pair used throughout is

    real space     u(r_j) = sum_G c_G exp(i G.r_j)     -> ifftn(c, norm="forward")
    reciprocal     c_G    = (1/N) sum_j u(r_j) e^-iGr  -> fftn(u, norm="forward")

Both directions therefore use ``norm="forward"``, which makes the transform pair exactly
inverse with no stray factors of N. Volume factors (the ``1/sqrt(Omega)`` in a normalised
Bloch state, the ``Omega/N`` of a real-space integral) are applied explicitly where they
belong rather than being hidden in the transform.

Grid size and aliasing
----------------------
A grid of ``n`` points per direction represents Miller indices ``-(n-1)/2 ... (n-1)/2``
without aliasing, so holding a sphere whose largest index is ``h_max`` needs
``n >= 2*h_max + 1``. Choosing a smaller grid does not raise an error, it silently folds
high-G components back onto low-G ones -- which is why the size is derived from the sphere
here rather than being a user setting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as _scipy_fft

from .gvectors import GSphere

__all__ = [
    "FFTGrid",
    "available_backends",
    "fast_grid_shape",
    "symmetry_compatible_multiple",
]

#: Radices scipy's pocketfft (and FFTW) handle efficiently.
_FAST_RADICES = (2, 3, 5, 7, 11)


def available_backends() -> tuple[str, ...]:
    """FFT backends usable in this environment (``pyfftw`` is an optional extra)."""
    backends = ["scipy"]
    try:  # pragma: no cover - depends on the optional extra being installed
        import pyfftw  # noqa: F401
    except ImportError:  # pragma: no cover
        pass
    else:  # pragma: no cover
        backends.append("pyfftw")
    return tuple(backends)


def fast_grid_shape(min_shape, multiple: int = 1) -> tuple[int, int, int]:
    """Round each dimension up to a length the FFT handles efficiently.

    ``multiple`` additionally forces each dimension to be divisible by that number, which
    is how a grid is made commensurate with the fractional translations of a space group
    (see :func:`symmetry_compatible_multiple`).

    >>> fast_grid_shape((17, 23, 31))
    (18, 24, 32)
    >>> fast_grid_shape((17, 23, 31), multiple=4)
    (20, 24, 32)
    """
    return tuple(int(_fast_length(int(n), multiple)) for n in min_shape)


def _fast_length(minimum: int, multiple: int) -> int:
    """Smallest efficient FFT length that is at least ``minimum`` and divides by ``multiple``."""
    if multiple <= 1:
        return int(_scipy_fft.next_fast_len(minimum))
    candidate = -(-minimum // multiple) * multiple
    for _ in range(1000):
        if _scipy_fft.next_fast_len(candidate) == candidate:
            return candidate
        candidate += multiple
    return int(_scipy_fft.next_fast_len(minimum))  # pragma: no cover - unreachable in practice


def symmetry_compatible_multiple(translations, maximum: int = 12) -> int:
    """Smallest grid divisor that turns every fractional translation into whole grid steps.

    A space group with a fractional translation ``f`` -- diamond's ``(1/4, 1/4, 1/4)``
    glide, say -- only maps the real-space grid onto itself if ``n f`` is an integer.
    When it is not, anything computed pointwise on the grid is *not* exactly invariant
    under the group, even though everything computed in reciprocal space still is. The
    exchange-correlation potential is precisely such a pointwise quantity, so a grid of 18
    divisions for diamond leaves a ``1e-10 Ha`` symmetry breaking in ``v_xc`` that a grid
    of 20 does not. It is small, but it is free to remove, and it will matter more for the
    forces and phonons of the later phases.

    Returns 1 when the group is symmorphic (all translations zero), and gives up (also
    returning 1) if no divisor up to ``maximum`` works, which happens only for translations
    that are not simple fractions.

    >>> symmetry_compatible_multiple([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    4
    >>> symmetry_compatible_multiple([[0.0, 0.0, 0.0]])
    1
    """
    import numpy as _np

    values = _np.asarray(translations, dtype=float).ravel()
    values = values - _np.floor(values)
    if values.size == 0 or _np.all(_np.abs(values) < 1e-8):
        return 1
    for multiple in range(1, maximum + 1):
        scaled = values * multiple
        if _np.all(_np.abs(scaled - _np.rint(scaled)) < 1e-6):
            return multiple
    return 1


@dataclass
class FFTGrid:
    """A real-space grid, with the index maps that move data to and from a G-sphere.

    ``shape`` is the number of grid points per lattice direction. The grid is uniform in
    fractional coordinates, so its real-space points are ``(i/n1, j/n2, k/n3)`` in the
    direct cell.
    """

    shape: tuple[int, int, int]
    #: Direct cell in Bohr (rows).
    cell: np.ndarray
    backend: str = "scipy"

    @property
    def size(self) -> int:
        """Total number of grid points."""
        return int(np.prod(self.shape))

    @property
    def volume(self) -> float:
        """Cell volume in Bohr^3."""
        return abs(float(np.linalg.det(self.cell)))

    @property
    def cell_volume_per_point(self) -> float:
        """Real-space integration weight ``Omega/N`` in Bohr^3."""
        return self.volume / self.size

    def __str__(self) -> str:
        n1, n2, n3 = self.shape
        return f"{n1}x{n2}x{n3} = {self.size} points ({self.backend} backend)"

    @classmethod
    def for_cutoff(cls, cell, ecut: float, *, backend: str = "scipy", multiple: int = 1) -> FFTGrid:
        """Smallest efficient grid that represents every ``|G|^2/2 <= ecut`` without aliasing.

        The sphere is built rather than bounded analytically. ``miller_bounds`` gives a
        rigorous box around the sphere, but for a non-orthogonal cell that box is loose:
        for the fcc primitive cell it yields 32^3 where the sphere actually needs 30^3, a
        21% larger transform for no benefit. Building the sphere costs one O(N) pass, done
        once per calculation.
        """
        sphere = GSphere.build(np.asarray(cell, dtype=float), ecut, sort=False)
        return cls.for_sphere(sphere, backend=backend, multiple=multiple)

    @classmethod
    def for_sphere(cls, sphere: GSphere, *, backend: str = "scipy", multiple: int = 1) -> FFTGrid:
        """Smallest efficient grid holding ``sphere`` without aliasing."""
        h_max = np.abs(sphere.miller).max(axis=0)
        return cls(
            shape=fast_grid_shape(2 * h_max + 1, multiple),
            cell=sphere.cell,
            backend=backend,
        )

    def indices(self, sphere: GSphere) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grid indices of each plane wave in ``sphere`` (negative indices wrap).

        Raises
        ------
        ValueError
            If a Miller index does not fit the grid, which would alias silently.
        """
        h_max = np.abs(sphere.miller).max(axis=0)
        if np.any(2 * h_max + 1 > np.asarray(self.shape)):
            needed = tuple(int(n) for n in 2 * h_max + 1)
            raise ValueError(
                f"grid {self.shape} is too small for Miller indices up to "
                f"{tuple(int(h) for h in h_max)}; at least {needed} is needed or "
                "high-G components alias onto low-G ones"
            )
        return tuple(sphere.miller[:, axis] % self.shape[axis] for axis in range(3))

    def scatter(self, coefficients: np.ndarray, sphere: GSphere) -> np.ndarray:
        """Place sphere coefficients onto a full grid, zero elsewhere.

        ``coefficients`` may be one vector of length ``npw`` or a batch of shape
        ``(nbands, npw)``; the batch axis is preserved.
        """
        i, j, k = self.indices(sphere)
        coefficients = np.asarray(coefficients)
        if coefficients.ndim == 1:
            grid = np.zeros(self.shape, dtype=complex)
            grid[i, j, k] = coefficients
            return grid
        grid = np.zeros((coefficients.shape[0], *self.shape), dtype=complex)
        grid[:, i, j, k] = coefficients
        return grid

    def gather(self, grid: np.ndarray, sphere: GSphere) -> np.ndarray:
        """Extract the sphere coefficients from a full grid (inverse of :meth:`scatter`)."""
        i, j, k = self.indices(sphere)
        grid = np.asarray(grid)
        return grid[i, j, k] if grid.ndim == 3 else grid[:, i, j, k]

    # -------------------------------------------------------------- transforms
    def _axes(self, array: np.ndarray) -> tuple[int, ...]:
        """The three spatial axes, which are the trailing ones for a batched array."""
        return (-3, -2, -1) if array.ndim >= 3 else tuple(range(array.ndim))

    def to_real(self, grid_reciprocal: np.ndarray) -> np.ndarray:
        """``c_G`` on the grid -> ``u(r) = sum_G c_G exp(i G.r)``."""
        array = np.asarray(grid_reciprocal)
        if self.backend == "pyfftw":  # pragma: no cover - optional extra
            import pyfftw.interfaces.numpy_fft as fftw

            return fftw.ifftn(array, axes=self._axes(array), norm="forward")
        return _scipy_fft.ifftn(array, axes=self._axes(array), norm="forward")

    def to_reciprocal(self, grid_real: np.ndarray) -> np.ndarray:
        """``u(r)`` on the grid -> ``c_G = (1/N) sum_r u(r) exp(-i G.r)``."""
        array = np.asarray(grid_real)
        if self.backend == "pyfftw":  # pragma: no cover - optional extra
            import pyfftw.interfaces.numpy_fft as fftw

            return fftw.fftn(array, axes=self._axes(array), norm="forward")
        return _scipy_fft.fftn(array, axes=self._axes(array), norm="forward")

    def wave_to_real(self, coefficients: np.ndarray, sphere: GSphere) -> np.ndarray:
        """Sphere coefficients -> real-space values, in one step."""
        return self.to_real(self.scatter(coefficients, sphere))

    def real_to_wave(self, values: np.ndarray, sphere: GSphere) -> np.ndarray:
        """Real-space values -> sphere coefficients, discarding components outside it."""
        return self.gather(self.to_reciprocal(values), sphere)

    def points_fractional(self) -> np.ndarray:
        """Fractional coordinates of the grid points, shape ``(*shape, 3)``."""
        axes = [np.arange(n) / n for n in self.shape]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def points_cartesian(self) -> np.ndarray:
        """Cartesian coordinates of the grid points in Bohr, shape ``(*shape, 3)``."""
        return self.points_fractional() @ self.cell

    def gvectors_cartesian(self, *, zero_nyquist: bool = False) -> np.ndarray:
        """Cartesian G-vectors of every grid point, in 1/Bohr, shape ``(*shape, 3)``.

        Frequencies follow the standard FFT ordering (``np.fft.fftfreq`` scaled by the
        grid size), so index ``i`` along an axis of length ``n`` carries Miller index
        ``i`` for ``i < n/2`` and ``i - n`` above that.

        ``zero_nyquist`` blanks the ``n/2`` component on each even-length axis. That
        component is its own negative on the grid, so an odd-order derivative built from
        it is not representable and would inject a spurious imaginary part into what must
        be a real field. Blanking it is the standard treatment for gradients; leave it
        alone for even-order operations such as the Laplacian.

        One consequence is worth knowing. The blanked set -- three planes of constant
        Miller index -- is not invariant under the point group of a non-orthogonal cell,
        so a quantity built from :meth:`gradient` or :meth:`divergence` is symmetric only
        to the extent that the field has no weight at the Nyquist planes. For a GGA
        potential, whose ``v_sigma`` factor is a nonlinear function of the density and so
        reaches every grid frequency, that leaves a symmetry breaking of order 1e-10 Ha
        on a grid barely large enough for the density -- and nothing measurable once the
        grid is comfortable. It is one more reason the density cutoff is set to four times
        the wavefunction cutoff rather than the bare minimum.
        """
        from .gvectors import reciprocal_cell

        axes = []
        for n in self.shape:
            frequencies = np.fft.fftfreq(n, d=1.0 / n)
            if zero_nyquist and n % 2 == 0:
                frequencies[n // 2] = 0.0
            axes.append(frequencies)
        miller = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return miller @ reciprocal_cell(self.cell)

    def gradient(self, values: np.ndarray) -> np.ndarray:
        """Gradient of a real periodic field, shape ``(*shape, 3)``, via ``i G`` in G space."""
        transformed = self.to_reciprocal(np.asarray(values, dtype=complex))
        gvectors = self.gvectors_cartesian(zero_nyquist=True)
        return np.stack(
            [self.to_real(1j * gvectors[..., axis] * transformed).real for axis in range(3)],
            axis=-1,
        )

    def divergence(self, field: np.ndarray) -> np.ndarray:
        """Divergence of a real vector field of shape ``(*shape, 3)``, via ``i G`` in G space."""
        gvectors = self.gvectors_cartesian(zero_nyquist=True)
        accumulated = np.zeros(self.shape, dtype=complex)
        for axis in range(3):
            transformed = self.to_reciprocal(np.asarray(field[..., axis], dtype=complex))
            accumulated += 1j * gvectors[..., axis] * transformed
        return self.to_real(accumulated).real

    def integrate(self, values: np.ndarray) -> complex:
        """Integrate a real-space function over the cell: ``(Omega/N) sum_r f(r)``."""
        return np.sum(values, axis=(-3, -2, -1)) * self.cell_volume_per_point
