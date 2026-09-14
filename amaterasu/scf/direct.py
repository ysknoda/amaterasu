"""Direct minimisation of the Kohn-Sham energy over the orbitals.

Density mixing solves the Kohn-Sham problem as a fixed point and can oscillate; direct
minimisation instead walks downhill on the energy itself, so every accepted step lowers
``E`` and there is nothing to diverge. The variables are the occupied orbitals subject to
``C C^dagger = I`` -- a Stiefel manifold -- and the machinery is the same L-BFGS or
conjugate-gradient method used for geometry optimisation, with two changes:

* the gradient is projected onto the constraint surface,
  ``H psi_n - sum_m psi_m <psi_m|H|psi_n>``, since a component inside the occupied
  subspace only rotates the orbitals among themselves and changes nothing;
* the step is followed by a *retraction* -- re-orthonormalising ``C + alpha D`` -- which
  brings the iterate back onto the manifold.

The Euclidean gradient is ``w_k g`` times that projected residual: the density dependence
of ``v_eff`` contributes nothing extra, by the same argument that turns the variational
condition into the Kohn-Sham equations.

The metric matters
------------------
Search directions are built from the *unweighted* residual and the inner product carries
the ``w_k g`` factors instead. Using the weighted residual as a direction with a single
step length would move a k-point of weight 0.002 five hundred times less far than one of
weight 0.1 -- the low-weight points would still be at their starting guess when the rest
had converged. Choosing the metric rather than the direction to carry the weights keeps
the step physically uniform while leaving the slope, and hence the line search, exact.

Scope
-----
Direct minimisation as written here needs **fixed integer occupations**, which is what
makes ``E`` a smooth function of the orbitals alone. With smearing the occupations are
variables too and the free energy has to be minimised over both, a genuinely different
algorithm; for a metal, ``scf.method: mixing`` is the supported route. This restriction is
shared with the direct minimisers in other plane-wave codes and is reported clearly
rather than worked around.

Symmetrisation is compatible with the gradient above: the group average is an orthogonal
projection, and a potential built from an already-symmetric density is symmetric, so
applying it to the residual changes nothing.
"""

from __future__ import annotations

import time

import numpy as np

from ..config.schema import Config
from ..hamiltonian.operator import Hamiltonian
from ..hamiltonian.system import PlaneWaveSystem
from ..units import EV_IN_HARTREE, HARTREE_IN_EV
from .density import compute_density, electron_count
from .eigen import davidson, orthonormalize, random_bands
from .eigen.precond import teter_payne_allan
from .occupations import compute_occupations

__all__ = ["channel_occupations", "run_direct_minimization"]


def channel_occupations(system: PlaneWaveSystem) -> list[int]:
    """Number of filled states per spin channel, or raise if they are not integers.

    For a spin-polarised run the split is fixed by the initial moment,
    ``N_up = (N + m)/2``: with fixed occupations there is no Fermi level to equalise the
    channels, so the moment is a constraint rather than an outcome.
    """
    total = system.nelectrons
    if system.nspin == 1:
        if abs(total / 2.0 - round(total / 2.0)) > 1e-9:
            raise ValueError(
                f"direct minimisation needs an even electron count, but the cell holds "
                f"{total:g}. Use scf.method: mixing with a smearing scheme."
            )
        return [round(total / 2.0)]

    moment = float(np.sum(system.initial_magmoms))
    counts = [(total + moment) / 2.0, (total - moment) / 2.0]
    for count in counts:
        if count < 0 or abs(count - round(count)) > 1e-9:
            raise ValueError(
                f"direct minimisation fixes the moment at the initial value "
                f"{moment:g} muB, which needs integer channel occupations; got "
                f"{counts[0]:g} up and {counts[1]:g} down. Use scf.method: mixing."
            )
    return [round(count) for count in counts]


def _occupation_array(system: PlaneWaveSystem, counts: list[int]) -> np.ndarray:
    occupations = np.zeros((system.nspin, system.nkpoints, max(counts)))
    for spin, count in enumerate(counts):
        occupations[spin, :, :count] = 1.0
    return occupations


def _energy_and_residual(system, orbitals, counts, *, gradient: bool = True):
    """Total energy at the current orbitals, and the projected (unweighted) residual."""
    from .driver import Energies, effective_potential

    occupations = _occupation_array(system, counts)
    density = compute_density(system, orbitals, occupations)
    potential, energy_hartree, energy_xc, _ = effective_potential(system, density)
    hamiltonian = Hamiltonian(system, potential)

    degeneracy = 2.0 / system.nspin
    kinetic = 0.0
    nonlocal_ = 0.0
    residual: list[list[np.ndarray]] = []
    for spin in range(system.nspin):
        per_spin = []
        for ik, sphere in enumerate(system.wave_spheres):
            block = orbitals[spin][ik]
            weight = degeneracy * system.kpoints.weights[ik]
            applied = hamiltonian.apply(block, ik, spin)
            if gradient:
                overlap = block.conj() @ applied.T
                per_spin.append(applied - overlap.T @ block)
            kinetic += weight * float(
                np.einsum("ng,g->", np.abs(block) ** 2, sphere.kinetic_energy)
            )
            nonlocal_ += system.projectors[ik].energy(block, np.full(len(block), weight))
        residual.append(per_spin)

    energies = Energies(
        kinetic=kinetic,
        local=float(np.sum(system.v_local * density.sum(axis=0)))
        * system.grid.cell_volume_per_point,
        hartree=energy_hartree,
        xc=energy_xc,
        nonlocal_=nonlocal_,
        ewald=system.ewald,
    )
    return energies, residual, density, occupations


def _weights(system: PlaneWaveSystem) -> np.ndarray:
    return (2.0 / system.nspin) * np.asarray(system.kpoints.weights, dtype=float)


def _dot(a, b, weights: np.ndarray) -> float:
    """``sum_k w_k g * 2 Re <a_k, b_k>``: the metric that differentiates the real energy."""
    total = 0.0
    for row_a, row_b in zip(a, b, strict=True):
        for weight, block_a, block_b in zip(weights, row_a, row_b, strict=True):
            total += weight * 2.0 * float(np.real(np.vdot(block_a, block_b)))
    return total


def _combine(a, b, scale: float):
    return [
        [x + scale * y for x, y in zip(row_a, row_b, strict=True)]
        for row_a, row_b in zip(a, b, strict=True)
    ]


def _negate(a):
    return [[-x for x in row] for row in a]


def _difference(a, b):
    return [
        [x - y for x, y in zip(row_a, row_b, strict=True)]
        for row_a, row_b in zip(a, b, strict=True)
    ]


def _retract(orbitals, direction, step):
    return [
        [orthonormalize(block + step * move) for block, move in zip(row, moves, strict=True)]
        for row, moves in zip(orbitals, direction, strict=True)
    ]


def _precondition(system, blocks, orbitals):
    """Apply the Teter-Payne-Allan preconditioner block by block."""
    out = []
    for spin, per_spin in enumerate(blocks):
        row = []
        for ik, block in enumerate(per_spin):
            sphere = system.wave_spheres[ik]
            reference = np.einsum("ng,g->n", np.abs(orbitals[spin][ik]) ** 2, sphere.kinetic_energy)
            row.append(teter_payne_allan(block, sphere.kinetic_energy, reference))
        out.append(row)
    return out


def _lbfgs_direction(system, residual, orbitals, steps, changes, weights):
    """Two-loop recursion with the TPA preconditioner as the initial inverse Hessian."""
    q = [[block.copy() for block in row] for row in residual]
    alphas = []
    rhos = []
    for step, change in zip(reversed(steps), reversed(changes), strict=True):
        curvature = _dot(change, step, weights)
        rho = 1.0 / curvature if abs(curvature) > 1e-30 else 0.0
        alpha = rho * _dot(step, q, weights)
        q = _combine(q, change, -alpha)
        alphas.append(alpha)
        rhos.append(rho)
    q = _precondition(system, q, orbitals)
    for (step, change), alpha, rho in zip(
        list(zip(steps, changes, strict=True)),
        list(reversed(alphas)),
        list(reversed(rhos)),
        strict=True,
    ):
        beta = rho * _dot(change, q, weights)
        q = _combine(q, step, alpha - beta)
    return _negate(q)


def _line_search(system, counts, orbitals, direction, slope, reference, trial):
    """Quadratic line search: one trial energy fixes the parabola through E(0) and E'(0).

    ``E(alpha)`` along a retracted path is not exactly a parabola, so the fitted minimum
    is clamped to a factor of four either side of the trial step and the result is only
    accepted if it actually lowers the energy; otherwise the trial is backtracked.
    """
    for _ in range(10):
        moved = _retract(orbitals, direction, trial)
        trial_energy = _energy_and_residual(system, moved, counts, gradient=False)[0].total
        curvature = trial_energy - reference - slope * trial
        if curvature > 0:
            step = float(np.clip(-0.5 * slope * trial**2 / curvature, 0.25 * trial, 4.0 * trial))
            fitted = _energy_and_residual(
                system, _retract(orbitals, direction, step), counts, gradient=False
            )[0].total
            if fitted <= min(reference, trial_energy):
                return step, fitted
        if trial_energy < reference:
            return trial, trial_energy
        trial *= 0.3
    return 0.0, reference


def run_direct_minimization(system: PlaneWaveSystem, config: Config, *, logger=None):
    """Minimise the Kohn-Sham energy over the occupied orbitals directly."""
    from .driver import SCFResult

    if system.is_paw:
        raise NotImplementedError(
            "direct minimisation is implemented for the Euclidean Stiefel manifold, where "
            "the orbitals are orthonormal. PAW makes them orthonormal in the S metric "
            "instead, which needs a different retraction and gradient. Use "
            "scf.method: mixing for a PAW calculation."
        )
    log = logger or (lambda *_: None)
    if config.occupations.scheme != "fixed":
        raise NotImplementedError(
            f"scf.method: {config.scf.method} minimises the energy over the orbitals at "
            f"fixed occupations, but occupations.scheme is "
            f"{config.occupations.scheme!r}. Set occupations.scheme: fixed for an "
            "insulator, or scf.method: mixing for a metal."
        )
    counts = channel_occupations(system)
    started = time.perf_counter()
    natoms = len(system.symbols)
    weights = _weights(system)
    use_lbfgs = config.scf.method == "direct-lbfgs"
    history_length = max(1, config.scf.lbfgs_history)
    threshold = config.scf.tol_energy * natoms * EV_IN_HARTREE

    orbitals = [
        [
            orthonormalize(
                random_bands(len(s), counts[spin], s.kinetic_energy, seed=97 * spin + ik)
            )
            for ik, s in enumerate(system.wave_spheres)
        ]
        for spin in range(system.nspin)
    ]

    steps: list = []
    changes: list = []
    previous = None
    last_step = None
    last_residual = None
    previous_energy = np.inf
    trial = 1.0
    scf_history: list[tuple[float, float, float]] = []
    converged = False
    iteration = 0
    energies = None
    density = None

    log(
        f"  {'iter':>4s} {'E (eV)':>18s} {'dE (eV/atom)':>14s} "
        f"{'|residual|':>13s} {'step':>9s} {'time (s)':>9s}"
    )
    for iteration in range(1, config.scf.max_steps + 1):
        step_started = time.perf_counter()
        energies, residual, density, _ = _energy_and_residual(system, orbitals, counts)
        energy = energies.total
        norm = np.sqrt(max(_dot(residual, residual, weights), 0.0))
        preconditioned = _precondition(system, residual, orbitals)

        # The secant pair for L-BFGS is completed here, where the new residual has just
        # been computed for its own sake -- forming it at the end of the previous pass
        # would evaluate the same Hamiltonian twice.
        if use_lbfgs and last_step is not None:
            steps.append(last_step)
            changes.append(_difference(residual, last_residual))
            if len(steps) > history_length:
                del steps[0]
                del changes[0]

        if use_lbfgs and steps:
            direction = _lbfgs_direction(system, residual, orbitals, steps, changes, weights)
        elif previous is not None:
            # Preconditioned Polak-Ribiere, restarted whenever beta would turn negative.
            denominator = _dot(previous["residual"], previous["preconditioned"], weights)
            numerator = _dot(_difference(residual, previous["residual"]), preconditioned, weights)
            beta = max(0.0, numerator / denominator) if abs(denominator) > 1e-30 else 0.0
            direction = _combine(_negate(preconditioned), previous["direction"], beta)
        else:
            direction = _negate(preconditioned)

        slope = _dot(residual, direction, weights)
        if slope >= 0:
            direction = _negate(preconditioned)
            slope = _dot(residual, direction, weights)
            steps.clear()
            changes.clear()
            last_step = None

        step, _ = _line_search(system, counts, orbitals, direction, slope, energy, trial)
        if step <= 0.0:  # pragma: no cover - the line search could not make progress
            log("  line search stalled; stopping")
            break
        trial = float(np.clip(step, 1e-3, 20.0))
        moved = _retract(orbitals, direction, step)

        delta = energy - previous_energy
        scf_history.append((energy, norm, 0.0))
        log(
            f"  {iteration:4d} {energy * HARTREE_IN_EV:18.9f} "
            f"{delta * HARTREE_IN_EV / natoms:14.2e} {norm:13.3e} {step:9.3f} "
            f"{time.perf_counter() - step_started:9.2f}"
        )
        if iteration > 1 and abs(delta) < threshold:
            converged = True
            break

        previous = {
            "residual": residual,
            "preconditioned": preconditioned,
            "direction": direction,
        }
        previous_energy = energy
        last_step = _difference(moved, orbitals)
        last_residual = residual
        orbitals = moved

    eigenvalues, wavefunctions = _final_diagonalization(system, density)
    occupation = compute_occupations(
        eigenvalues,
        system.kpoints.weights,
        system.nelectrons,
        scheme="fixed",
        nspin=system.nspin,
    )
    return SCFResult(
        converged=converged,
        iterations=iteration,
        energies=energies,
        eigenvalues=eigenvalues,
        occupations=occupation.occupations,
        fermi_level=occupation.fermi_level,
        gap=occupation.gap,
        magnetization=occupation.magnetization,
        density=density,
        wavefunctions=wavefunctions,
        history=scf_history,
        electron_count=electron_count(system, density),
        seconds=time.perf_counter() - started,
    )


def _final_diagonalization(system, density):
    """Diagonalise the converged Hamiltonian for the full band set.

    Direct minimisation never forms eigenvalues -- it only needs the occupied subspace --
    so the empty states and the band energies come from one diagonalisation at the end.
    """
    from .driver import effective_potential

    potential, _, _, _ = effective_potential(system, density)
    hamiltonian = Hamiltonian(system, potential)
    eigenvalues = np.empty((system.nspin, system.nkpoints, system.nbands))
    wavefunctions = []
    for spin in range(system.nspin):
        per_spin = []
        for ik, sphere in enumerate(system.wave_spheres):
            result = davidson(
                lambda block, ik=ik, spin=spin: hamiltonian.apply(block, ik, spin),
                random_bands(len(sphere), system.nbands, sphere.kinetic_energy, seed=7 + ik),
                sphere.kinetic_energy,
                tolerance=1e-9,
                max_iterations=80,
            )
            eigenvalues[spin, ik] = result.eigenvalues
            per_spin.append(result.coefficients)
        wavefunctions.append(per_spin)
    return eigenvalues, wavefunctions
