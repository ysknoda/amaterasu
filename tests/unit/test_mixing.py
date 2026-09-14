"""Density mixing: convergence acceleration and the charge conservation it must not break."""

from __future__ import annotations

import numpy as np
import pytest

from amaterasu.basis.fft import FFTGrid
from amaterasu.basis.gvectors import GSphere
from amaterasu.config.schema import MixingConfig
from amaterasu.scf.mixing import (
    AndersonMixer,
    BroydenMixer,
    KerkerPreconditioner,
    LinearMixer,
    PulayMixer,
    create_mixer,
)


def _linear_fixed_point(mixer, size=40, radius=0.95, seed=1, steps=300):
    """Iterate ``n_out = A n_in + b`` and report how close the mixer gets in ``steps``."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(size, size))
    matrix = matrix / np.abs(np.linalg.eigvals(matrix)).max() * radius
    offset = rng.normal(size=size)
    exact = np.linalg.solve(np.eye(size) - matrix, offset)

    current = np.zeros((1, size, 1, 1))
    for _ in range(steps):
        output = (matrix @ current.ravel() + offset).reshape(current.shape)
        if np.linalg.norm(output - current) < 1e-13:
            break
        current = mixer(current, output)
    return float(np.linalg.norm(current.ravel() - exact))


def test_history_mixers_beat_linear_mixing():
    """The whole point of Pulay and Broyden: a fixed point that linear mixing cannot reach.

    With a spectral radius of 0.95 the plain iteration needs thousands of steps, while a
    quasi-Newton history converges in a few hundred.
    """
    linear = _linear_fixed_point(LinearMixer(beta=0.3))
    pulay = _linear_fixed_point(PulayMixer(beta=0.3, history=8))
    broyden = _linear_fixed_point(BroydenMixer(beta=0.3, history=8))
    anderson = _linear_fixed_point(AndersonMixer(beta=0.3))
    assert pulay < 1e-9
    assert broyden < 1e-9
    assert anderson < linear
    # Linear mixing is still crawling after the same number of steps, by many orders.
    assert linear > 1e5 * max(pulay, broyden)


def test_a_fixed_point_is_left_alone():
    """If the output equals the input, every mixer must return that same density."""
    density = np.abs(np.random.default_rng(0).normal(size=(1, 4, 4, 4)))
    for mixer in (LinearMixer(), PulayMixer(), BroydenMixer(), AndersonMixer()):
        assert mixer(density, density) == pytest.approx(density, abs=1e-12)


def test_kerker_leaves_the_magnetisation_alone(si_atoms):
    """The charge channel is preconditioned; the spin channel passes through untouched.

    Kerker inverts the *charge* response of an electron gas, which diverges as 1/q^2
    because charge is screened. Magnetization is not screened, and since the factor is
    exactly zero at G = 0, applying it to the spin channel would freeze the total moment
    -- the mixer could never change it at all.
    """
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 20.0)
    preconditioner = KerkerPreconditioner(grid, q0=1.5)

    rng = np.random.default_rng(2)
    residual = rng.normal(size=(2, *grid.shape))
    filtered = preconditioner(residual)
    # The charge channel loses its mean; the spin channel keeps its own exactly.
    assert abs(float(np.sum(filtered[0] + filtered[1]))) < 1e-9 * abs(
        float(np.sum(residual[0] + residual[1]))
    )
    assert filtered[0] - filtered[1] == pytest.approx(residual[0] - residual[1], abs=1e-12)


def test_kerker_conserves_charge_exactly(si_atoms):
    """The G = 0 factor is exactly zero, so a preconditioned residual integrates to zero.

    That is what keeps the electron count fixed through the SCF no matter how large a
    step the mixer takes.
    """
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 20.0)
    preconditioner = KerkerPreconditioner(grid, q0=1.5)

    rng = np.random.default_rng(0)
    residual = rng.normal(size=(1, *grid.shape))
    residual += 0.7  # a deliberately non-zero mean, i.e. a charge-changing residual
    filtered = preconditioner(residual)
    assert abs(float(np.sum(filtered))) < 1e-9 * abs(float(np.sum(residual)))


def test_kerker_damps_long_wavelengths_more_than_short(si_atoms):
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 20.0)
    sphere = GSphere.build(cell, 20.0)
    preconditioner = KerkerPreconditioner(grid, q0=1.5)

    order = np.argsort(sphere.g2)
    lowest, highest = order[1], order[-1]
    factors = []
    for index in (lowest, highest):
        coefficients = np.zeros(len(sphere), dtype=complex)
        coefficients[index] = 1.0
        field = grid.to_real(grid.scatter(coefficients, sphere)).real[None, ...]
        factors.append(float(np.max(np.abs(preconditioner(field)))) / float(np.max(np.abs(field))))
    # Each Fourier component is scaled by exactly G^2/(G^2 + q0^2).
    expected = sphere.g2[[lowest, highest]] / (sphere.g2[[lowest, highest]] + 1.5**2)
    assert factors == pytest.approx(expected, rel=1e-9)
    assert factors[0] < 0.5 < factors[1]


def test_kerker_is_a_no_op_when_disabled(si_atoms):
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 20.0)
    field = np.ones((1, *grid.shape))
    assert KerkerPreconditioner(grid, q0=0.0)(field) is field


def test_history_is_bounded():
    mixer = PulayMixer(beta=0.4, history=3)
    rng = np.random.default_rng(0)
    for _ in range(10):
        density = rng.normal(size=(1, 2, 2, 2))
        mixer(density, density + 0.01 * rng.normal(size=density.shape))
    assert len(mixer.inputs) <= 3
    mixer.reset()
    assert mixer.inputs == [] and mixer.residuals == []


@pytest.mark.parametrize("algorithm", ["linear", "anderson", "pulay", "broyden"])
def test_create_mixer_attaches_the_preconditioner(si_atoms, algorithm):
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 12.0)
    mixer = create_mixer(MixingConfig(algorithm=algorithm), grid)
    assert isinstance(mixer.precondition, KerkerPreconditioner)
    assert mixer.beta == 0.4


def test_kerker_preserves_components_outside_the_cutoff_sphere(si_atoms):
    """A preconditioner may scale a component; it must never delete one.

    An earlier version filtered on the density sphere and scattered back, which zeroed
    everything outside it. The density floor in the starting guess does put a little
    weight there, and the mixer could then never remove that part of the residual: the
    SCF converged in energy while the density residual sat at a constant value forever,
    a milli-electronvolt away from the true solution.
    """
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 20.0)
    sphere = GSphere.build(cell, 20.0)
    preconditioner = KerkerPreconditioner(grid, q0=1.5)

    # White noise has weight at every frequency the grid carries, including the corners
    # of the box that lie outside the sphere; keep only that part.
    noise = np.random.default_rng(0).normal(size=grid.shape)
    inside = grid.gather(grid.to_reciprocal(noise.astype(complex)), sphere)
    outside = noise - grid.to_real(grid.scatter(inside, sphere)).real
    assert np.sqrt(np.mean(outside**2)) > 1e-6, "the test field must have out-of-sphere weight"

    filtered = preconditioner(outside[None, ...])[0]
    # High-frequency components pass essentially unchanged, and certainly not to zero.
    assert np.sqrt(np.mean(filtered**2)) > 0.5 * np.sqrt(np.mean(outside**2))


def test_kerker_lets_the_charge_float_when_asked(si_atoms):
    """Under PAW the total smooth charge is not a constant, and must survive the filter.

    ``N - sum_ij D_ij O_ij`` moves as the occupancy matrices do, so pinning ``G = 0`` --
    right for a norm-conserving run, see the test above -- leaves the mixer unable to
    reach the fixed point at all. Everything above ``G = 0`` must be damped exactly as
    before: this is one component changing, not a different preconditioner.
    """
    from amaterasu.units import ANG_IN_BOHR

    cell = np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR
    grid = FFTGrid.for_cutoff(cell, 20.0)
    rng = np.random.default_rng(0)
    residual = rng.normal(size=(1, *grid.shape)) + 0.7

    pinned = KerkerPreconditioner(grid, q0=1.5)(residual)
    floating = KerkerPreconditioner(grid, q0=1.5, uniform_factor=1.0)(residual)

    assert float(np.sum(floating)) == pytest.approx(float(np.sum(residual)), rel=1e-12)
    # The two differ by a constant and nothing else.
    difference = floating - pinned
    assert difference.std() < 1e-9 * abs(difference.mean())


def test_create_mixer_floats_the_charge_only_for_paw(si_atoms):
    from amaterasu.units import ANG_IN_BOHR

    grid = FFTGrid.for_cutoff(np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR, 20.0)
    config = MixingConfig(algorithm="pulay", preconditioner="kerker")
    assert create_mixer(config, grid).precondition.uniform_factor == 0.0
    assert create_mixer(config, grid, floating_charge=True).precondition.uniform_factor == 1.0


def test_block_preconditioner_filters_only_the_leading_block(si_atoms):
    """PAW packs the density and the occupancy matrices into one vector to mix them.

    Only the density half lives on a grid, so only that half may be Kerker-filtered; the
    occupancy residual has to arrive on the other side untouched. Mixing the two with
    separate schemes -- Pulay for the density, a damped linear step for ``D_ij`` -- makes
    them describe different iterates, and the density residual floors out three orders of
    magnitude above the threshold instead of converging.
    """
    from amaterasu.scf.mixing import BlockPreconditioner
    from amaterasu.units import ANG_IN_BOHR

    grid = FFTGrid.for_cutoff(np.asarray(si_atoms.cell[:]) * ANG_IN_BOHR, 20.0)
    shape = (1, *grid.shape)
    rng = np.random.default_rng(1)
    density_part = rng.normal(size=shape)
    tail = rng.normal(size=17)
    packed = np.concatenate([density_part.ravel(), tail])

    inner = KerkerPreconditioner(grid, q0=1.5)
    filtered = BlockPreconditioner(inner, shape)(packed)

    assert filtered[density_part.size :] == pytest.approx(tail, rel=1e-15)
    assert filtered[: density_part.size] == pytest.approx(inner(density_part).ravel(), rel=1e-15)
