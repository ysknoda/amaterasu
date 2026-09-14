"""Density mixing: turning the SCF fixed-point map into a convergent iteration.

Feeding the output density straight back as the next input diverges for anything but a
small molecule. The reason is charge sloshing: a small change in the potential at long
wavelength moves a large amount of charge, because the electron gas responds with a
dielectric function that behaves as ``1 + q_TF^2/q^2`` and so diverges as ``q -> 0``.

Two devices fix it.

**Kerker preconditioning** multiplies the residual by ``G^2 / (G^2 + q0^2)``
(Kerker, PRB 23, 3082 (1981)), which is the inverse of that model response: it damps the
long-wavelength components that would otherwise slosh, and leaves short wavelengths
alone. At ``G = 0`` the factor is exactly zero, so the electron count is conserved to
machine precision no matter what the mixer does -- which is right whenever the density
holds every valence electron, and wrong under PAW, where it does not; see
``uniform_factor``.

**History extrapolation** (Pulay's DIIS, and Broyden's quasi-Newton update) uses the
sequence of past inputs and residuals to build an approximate inverse Jacobian, replacing
the fixed scalar step of linear mixing.

All mixers here operate on the real-space density of shape ``(nspin, *grid.shape)``,
flattened; the preconditioner is the only part that visits reciprocal space.

References: Pulay, Chem. Phys. Lett. 73, 393 (1980); Johnson, PRB 38, 12807 (1988).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..basis.fft import FFTGrid
from ..config.schema import MixingConfig

__all__ = [
    "AndersonMixer",
    "BlockPreconditioner",
    "BroydenMixer",
    "LinearMixer",
    "Mixer",
    "PulayMixer",
    "create_mixer",
]


@dataclass
class KerkerPreconditioner:
    """``G^2 / (G^2 + q0^2)`` applied to the charge residual, but never to the spin residual.

    Kerker's factor is the inverse of the *charge* response of an electron gas, which
    diverges as ``1/q^2`` because a long-wavelength charge fluctuation is screened. The
    magnetization has no such divergence: it is not screened, so damping its long
    wavelengths is a physical error, not a stabilisation.

    The consequence is not subtle. The factor is exactly zero at ``G = 0``, so applying it
    to the magnetization would stop the mixer from ever changing the *total moment* --
    which could then only drift through the occupations, one small step per iteration. For
    ferromagnetic iron started at 2.2 muB this leaves the moment crawling and the density
    residual stuck three orders of magnitude above the threshold. Preconditioning the
    charge channel alone is what every production code does, and it is what makes iron
    converge here.

    The filter runs over *every* frequency the grid carries, not just the cutoff sphere.
    That distinction matters: a preconditioner may scale a component, but it must never
    delete one. Restricting it to the sphere silently zeroes whatever lies outside --
    which the density floor in the starting guess does produce -- and the mixer can then
    never remove that part of the residual. It shows up as an SCF that converges in energy
    while the density residual sits at a constant value forever, and settles about a
    milli-electronvolt away from the true solution.
    """

    grid: FFTGrid
    q0: float
    #: What to do with ``G = 0``. Zero -- the norm-conserving case -- pins the total
    #: charge, which is right when the density always holds every valence electron. Under
    #: PAW it does not: the smooth charge is ``N - sum_ij D_ij O_ij`` and follows the
    #: occupancy matrices, so pinning it makes the residual unable to reach zero. Passing
    #: it through at 1.0 lets the mixer move that one component with its ordinary history
    #: and damping, which is what a preconditioner should do -- scale a component, never
    #: delete it. Correcting the charge *after* mixing instead leaves the mixer's stored
    #: input/output pairs describing a vector that was never used, and a history mixer
    #: fed an inconsistent history extrapolates badly: on compressed silicon Pulay and
    #: Broyden both stalled thirty times worse than plain linear mixing.
    uniform_factor: float = 0.0

    def __post_init__(self):
        gvectors = self.grid.gvectors_cartesian()
        g2 = np.einsum("...i,...i->...", gvectors, gvectors)
        with np.errstate(invalid="ignore"):
            self._factor = np.where(g2 > 1e-12, g2 / (g2 + self.q0**2), float(self.uniform_factor))

    def _filter(self, field: np.ndarray) -> np.ndarray:
        transformed = self.grid.to_reciprocal(field.astype(complex))
        return self.grid.to_real(transformed * self._factor).real

    def __call__(self, residual: np.ndarray) -> np.ndarray:
        if self.q0 <= 0:
            return residual
        if residual.shape[0] == 1:
            return self._filter(residual[0])[None, ...]
        charge = self._filter(residual[0] + residual[1])
        moment = residual[0] - residual[1]
        return np.stack([0.5 * (charge + moment), 0.5 * (charge - moment)])


@dataclass
class BlockPreconditioner:
    """Apply ``inner`` to the leading block of a packed residual and pass the rest through.

    PAW has two things to mix, not one: the smooth density and the atomic occupancy
    matrices ``D_ij``. Both are functionals of the same orbitals, so they have to travel
    through the same history -- mixing the density with Pulay while damping ``D`` with a
    plain linear step leaves the two describing different iterates, and the residual
    floors out around ``1e-3`` instead of converging. Packing them into one vector fixes
    that, but only the density half lives on a grid and can be Kerker-filtered.
    """

    inner: KerkerPreconditioner
    shape: tuple[int, ...]

    def __post_init__(self):
        self.size = int(np.prod(self.shape))

    def __call__(self, residual: np.ndarray) -> np.ndarray:
        head = self.inner(residual[: self.size].reshape(self.shape)).ravel()
        return np.concatenate([head, residual[self.size :]])


@dataclass
class Mixer:
    """Base mixer: plain linear mixing, and the storage the history mixers extend."""

    beta: float = 0.4
    history: int = 8
    precondition = staticmethod(lambda residual: residual)
    inputs: list[np.ndarray] = field(default_factory=list)
    residuals: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.inputs.clear()
        self.residuals.clear()

    def _record(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        residual = density_out - density_in
        self.inputs.append(density_in.copy())
        self.residuals.append(residual)
        if len(self.inputs) > self.history:
            del self.inputs[0]
            del self.residuals[0]
        return residual

    def __call__(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        residual = self._record(density_in, density_out)
        return density_in + self.beta * self.precondition(residual)


class LinearMixer(Mixer):
    """``n_new = n_in + beta * P(n_out - n_in)``. Robust, and slow for anything metallic."""


class PulayMixer(Mixer):
    """Pulay/DIIS: minimise the norm of a linear combination of past residuals.

    The coefficients solve ``min || sum_i c_i R_i ||`` subject to ``sum_i c_i = 1``, which
    is the least-squares statement that the extrapolated point is a fixed point. The
    Lagrange-multiplier system is bordered with a row and column of ones; it becomes
    singular once the residuals are linearly dependent, which is the normal state of
    affairs near convergence, so the oldest entries are dropped until it is solvable.
    """

    def _coefficients(self) -> np.ndarray:
        while True:
            size = len(self.residuals)
            if size == 1:
                return np.ones(1)
            flat = np.array([r.ravel() for r in self.residuals])
            overlap = flat @ flat.T
            scale = float(np.trace(overlap)) / size
            matrix = np.ones((size + 1, size + 1))
            matrix[:size, :size] = overlap / scale
            matrix[size, size] = 0.0
            target = np.zeros(size + 1)
            target[size] = 1.0
            try:
                solution = np.linalg.solve(matrix, target)
            except np.linalg.LinAlgError:
                pass
            else:
                if np.all(np.isfinite(solution)) and np.max(np.abs(solution[:size])) < 1e4:
                    return solution[:size]
            del self.inputs[0]
            del self.residuals[0]

    def __call__(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        self._record(density_in, density_out)
        coefficients = self._coefficients()
        mixed_in = np.einsum("i,i...->...", coefficients, np.array(self.inputs))
        mixed_residual = np.einsum("i,i...->...", coefficients, np.array(self.residuals))
        return mixed_in + self.beta * self.precondition(mixed_residual)


class AndersonMixer(PulayMixer):
    """Anderson mixing: Pulay restricted to the two most recent iterations.

    Kept as a separate name because it is the classical two-term form quoted in the
    literature, and because a short history is sometimes the more stable choice when the
    residuals are noisy.
    """

    def __init__(self, beta: float = 0.4, history: int = 2, **kwargs):
        super().__init__(beta=beta, history=min(history, 2), **kwargs)


@dataclass
class BroydenMixer(Mixer):
    """Johnson's modified Broyden update (PRB 38, 12807 (1988)).

    Broyden builds an approximate inverse Jacobian from the observed
    ``(delta n, delta residual)`` pairs instead of assuming a scalar one. ``w0`` weights
    the "stay close to the previous Jacobian" constraint against the secant conditions and
    regularises the normal equations; Johnson's 0.01 is used throughout the literature.
    """

    w0: float = 0.01
    weight: float = 1.0

    def __call__(self, density_in: np.ndarray, density_out: np.ndarray) -> np.ndarray:
        residual = self._record(density_in, density_out)
        step = self.beta * self.precondition(residual)
        if len(self.residuals) < 2:
            return density_in + step

        # Secant pairs, normalised by the residual change as Johnson prescribes.
        delta_residual = []
        delta_input = []
        for i in range(len(self.residuals) - 1):
            difference = self.residuals[i + 1] - self.residuals[i]
            norm = float(np.linalg.norm(difference))
            if norm < 1e-30:  # pragma: no cover - identical iterates
                continue
            delta_residual.append(difference.ravel() / norm)
            delta_input.append(
                (
                    self.inputs[i + 1] - self.inputs[i] + self.beta * self.precondition(difference)
                ).ravel()
                / norm
            )
        if not delta_residual:  # pragma: no cover - defensive
            return density_in + step

        df = np.array(delta_residual)
        dn = np.array(delta_input)
        size = len(df)
        matrix = self.weight**2 * (df @ df.T) + self.w0**2 * np.eye(size)
        overlaps = self.weight * (df @ residual.ravel())
        gamma = np.linalg.solve(matrix, overlaps) * self.weight
        correction = np.einsum("i,i...->...", gamma, dn).reshape(density_in.shape)
        return density_in + step - correction


_MIXERS = {
    "linear": LinearMixer,
    "anderson": AndersonMixer,
    "pulay": PulayMixer,
    "broyden": BroydenMixer,
}


def create_mixer(config: MixingConfig, grid: FFTGrid, *, floating_charge: bool = False) -> Mixer:
    """Build the mixer named in ``scf.mixing`` with its preconditioner attached.

    ``floating_charge`` says the density's total charge is not a constant of the
    calculation, which is the PAW case; see :class:`KerkerPreconditioner`.
    """
    factory = _MIXERS.get(config.algorithm)
    if factory is None:  # pragma: no cover - the schema restricts the value
        raise ValueError(f"unknown mixing algorithm {config.algorithm!r}")
    mixer = factory(beta=config.beta, history=config.history)
    if config.preconditioner == "kerker":
        mixer.precondition = KerkerPreconditioner(
            grid, config.kerker_q0, uniform_factor=1.0 if floating_charge else 0.0
        )
    return mixer
