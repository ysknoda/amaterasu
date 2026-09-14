"""The self-consistent field loop.

Kohn-Sham theory is a fixed-point problem: the potential depends on the density, the
density on the orbitals, and the orbitals on the potential. One pass of the loop is

    n  ->  v_eff[n]  ->  diagonalise H  ->  occupy  ->  n_out ,

and the mixer in :mod:`amaterasu.scf.mixing` decides what to feed back.

Which energy is reported
------------------------
The total energy is the Kohn-Sham functional evaluated at the *output* density and the
current orbitals,

    E = T[psi] + E_NL[psi] + integral v_loc n_out + E_H[n_out] + E_xc[n_out + n_core]
        + E_Ewald ,

so it is the variational energy of the orbitals actually in hand rather than a mixture of
input and output quantities. With smearing the variational quantity is the free energy
``F = E - sigma S``; that is what the convergence test watches, and what the forces of
Phase 3 will be the gradient of.

Two identities are checked every iteration rather than only at the end, because each one
localises a different class of bug:

* ``sum_nk w f eps == T + integral v_eff n_out + E_NL`` -- ties the eigenvalues, the
  Hamiltonian and the density together. It fails if the operator and the density
  accumulation disagree about normalisation, spin degeneracy or the k-point weights.
* ``integral n_out = N_electrons`` -- fails if the sphere-to-grid mapping loses
  coefficients or the symmetrisation is inconsistent with the k-point set.

Convergence is judged on the energy change per atom *and* the density change per valence
electron, both of which must be met. The energy alone is not enough: it is stationary at
the solution, so it stops moving while the density is still drifting.

Both are reported in the result, not merely asserted, so a marginal case shows up as a
number in the log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config.schema import Config
from ..hamiltonian.hartree import hartree
from ..hamiltonian.operator import Hamiltonian
from ..hamiltonian.system import PlaneWaveSystem
from ..pseudo.paw.onsite import OnsiteResult
from ..units import EV_IN_HARTREE, HARTREE_IN_EV
from .density import compute_density, electron_count
from .eigen import random_bands, solve
from .mixing import BlockPreconditioner, KerkerPreconditioner, create_mixer
from .occupations import compute_occupations

__all__ = ["Energies", "PAWState", "SCFResult", "effective_potential", "paw_state", "run_scf"]


@dataclass
class Energies:
    """Total-energy decomposition in Hartree."""

    kinetic: float = 0.0
    local: float = 0.0
    hartree: float = 0.0
    xc: float = 0.0
    nonlocal_: float = 0.0
    ewald: float = 0.0
    entropy: float = 0.0

    @property
    def total(self) -> float:
        """The Kohn-Sham energy ``E`` (no entropy term)."""
        return self.kinetic + self.local + self.hartree + self.xc + self.nonlocal_ + self.ewald

    @property
    def free_energy(self) -> float:
        """``F = E - sigma S``: the variational quantity when the occupations are smeared."""
        return self.total - self.entropy

    @property
    def extrapolated(self) -> float:
        """``E0 = (E + F)/2``, the ``sigma -> 0`` estimate for a symmetric smearing."""
        return self.total - 0.5 * self.entropy

    def render(self, natoms: int) -> str:
        """A table in the output units (eV), with the per-atom total."""
        rows = [
            ("kinetic", self.kinetic),
            ("local pseudopotential", self.local),
            ("Hartree", self.hartree),
            ("exchange-correlation", self.xc),
            ("nonlocal pseudopotential", self.nonlocal_),
            ("Ewald (ion-ion)", self.ewald),
            ("-T*S (smearing)", -self.entropy),
        ]
        lines = [f"  {name:<26s} {value * HARTREE_IN_EV:16.9f} eV" for name, value in rows]
        lines.append("  " + "-" * 45)
        lines.append(f"  {'total energy E':<26s} {self.total * HARTREE_IN_EV:16.9f} eV")
        lines.append(
            f"  {'free energy F = E - TS':<26s} {self.free_energy * HARTREE_IN_EV:16.9f} eV"
        )
        lines.append(
            f"  {'extrapolated E0':<26s} {self.extrapolated * HARTREE_IN_EV:16.9f} eV"
            f"   ({self.extrapolated * HARTREE_IN_EV / natoms:.9f} eV/atom)"
        )
        return "\n".join(lines)


@dataclass
class SCFResult:
    """Everything a converged (or abandoned) SCF cycle produced."""

    converged: bool
    iterations: int
    energies: Energies
    #: Shape ``(nspin, nkpoints, nbands)`` in Hartree.
    eigenvalues: np.ndarray
    occupations: np.ndarray
    fermi_level: float
    gap: float | None
    magnetization: float
    #: Shape ``(nspin, *grid.shape)`` in electrons/Bohr^3.
    density: np.ndarray
    #: ``wavefunctions[spin][ik]``, shape ``(nbands, npw_k)``.
    wavefunctions: list[list[np.ndarray]]
    #: Per-iteration ``(free energy in Hartree, density residual per electron,
    #: eigenvalue-sum residual in Hartree)``.
    history: list[tuple[float, float, float]] = field(default_factory=list)
    electron_count: float = 0.0
    seconds: float = 0.0

    @property
    def homo_lumo(self) -> tuple[float, float] | None:
        """Highest occupied and lowest unoccupied eigenvalue in Hartree, if there is a gap."""
        if self.gap is None:
            return None
        occupied = self.eigenvalues[self.occupations > 0.5]
        empty = self.eigenvalues[self.occupations <= 0.5]
        if occupied.size == 0 or empty.size == 0:  # pragma: no cover - defensive
            return None
        return float(occupied.max()), float(empty.min())


@dataclass
class PAWState:
    """The PAW quantities that depend on the atomic occupancy matrices.

    These sit between the two halves of a PAW calculation. The occupancies come from the
    orbitals, the one-centre corrections and the compensation charge come from them, and
    the compensation charge then changes the smooth Hartree potential, which feeds back
    into the coupling ``H^a_ij``. There is no circularity: each step depends only on the
    one before it.
    """

    #: ``D_ij`` for the whole cell, block-diagonal over atoms, ``(nspin, nproj, nproj)``.
    occupancies: np.ndarray
    onsite: list[OnsiteResult]
    #: The compensation charge on the real-space grid, in electrons/Bohr^3.
    compensation: np.ndarray
    #: ``sum_a (E^a - E~^a)`` in Hartree.
    energy: float
    #: ``H^a_ij = dE/dD_ij``, filled in once the smooth Hartree potential is known.
    coupling: np.ndarray | None = None

    def augmented_charge(self, system: PlaneWaveSystem) -> float:
        """``sum_a sum_ij D_ij O_ij``: the charge the smooth orbitals are missing."""
        return float(np.einsum("sij,ij->", self.occupancies, system.paw.overlap))


def paw_state(system: PlaneWaveSystem, occupancies: np.ndarray) -> PAWState:
    """One-centre corrections and the compensation charge for a set of occupancies."""
    paw = system.paw
    per_atom = paw.occupancies(occupancies)
    onsite = [
        paw.setups[symbol].compute(block)
        for symbol, block in zip(system.symbols, per_atom, strict=True)
    ]
    coefficients = paw.compensation_charge([result.multipoles for result in onsite])
    compensation = system.grid.to_real(
        system.grid.scatter(coefficients, system.density_sphere)
    ).real
    return PAWState(
        occupancies=occupancies,
        onsite=onsite,
        compensation=np.ascontiguousarray(compensation),
        energy=float(sum(result.energy for result in onsite)),
    )


def _hartree_charge(
    system: PlaneWaveSystem, density: np.ndarray, state: PAWState | None
) -> np.ndarray:
    """The charge density the Hartree term sees.

    For a norm-conserving run that is just the valence density: the partial core is an
    exchange-correlation device and carries no charge. For PAW the smooth core *is* part
    of the charge, and the compensation charge carries the nuclei, which together make the
    density neutral and let the plane-wave Hartree energy stand in for the whole of the
    electrostatics -- Ewald sum included.
    """
    total = density.sum(axis=0)
    if state is None:
        return total
    if system.core_density is not None:
        total = total + system.core_density
    return total + state.compensation


def _local_charge(system: PlaneWaveSystem, density: np.ndarray) -> np.ndarray:
    """The density the local ionic term multiplies.

    The PAW zero potential couples to ``n~``, which includes the smooth core (Eq. 18 of
    Mortensen et al.); the norm-conserving local pseudopotential couples to the valence
    density alone.
    """
    total = density.sum(axis=0)
    if system.is_paw and system.core_density is not None:
        total = total + system.core_density
    return total


def effective_potential(
    system: PlaneWaveSystem, density: np.ndarray, state: PAWState | None = None
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Build ``v_eff``; return ``(v_eff, E_Hartree, E_xc, v_Hartree on the sphere)``.

    The partial core density enters the exchange-correlation functional in both methods.
    In a norm-conserving run that is *all* it does -- never the Hartree term and never the
    electron count -- which is the whole content of the nonlinear core correction. A PAW
    smooth core is additionally a real charge; see :func:`_hartree_charge`.

    The Hartree potential is returned on the density sphere as well as folded into
    ``v_eff``, because PAW needs its projection onto the compensation shape functions and
    recomputing the transform would be wasteful.
    """
    energy_hartree, v_hartree = hartree(
        system.grid, system.density_sphere, _hartree_charge(system, density, state)
    )

    xc_density = density
    if system.core_density is not None:
        xc_density = density + system.core_density[None, ...] / system.nspin
    xc = system.xc.evaluate_on_grid(xc_density, system.grid)

    potential = system.v_local[None, ...] + v_hartree[None, ...] + xc.potential
    hartree_g = system.grid.gather(
        system.grid.to_reciprocal(v_hartree.astype(complex)), system.density_sphere
    )
    return np.ascontiguousarray(potential), energy_hartree, xc.energy, hartree_g


def _initial_wavefunctions(system: PlaneWaveSystem, seed: int) -> list[list[np.ndarray]]:
    return [
        [
            random_bands(
                len(sphere), system.nbands, sphere.kinetic_energy, seed=seed + 991 * spin + ik
            )
            for ik, sphere in enumerate(system.wave_spheres)
        ]
        for spin in range(system.nspin)
    ]


def run_scf(
    system: PlaneWaveSystem,
    config: Config,
    *,
    logger=None,
    wavefunctions: list[list[np.ndarray]] | None = None,
    density: np.ndarray | None = None,
) -> SCFResult:
    """Run the SCF loop to self-consistency.

    ``wavefunctions`` and ``density`` allow restarting from a previous result, which is
    what the geometry optimiser and molecular dynamics of Phase 3 will do at every step.
    Both must belong to the same basis: the same grid, the same k-points and the same band
    count. :func:`amaterasu.solvers.scf.run` checks that before passing them in.
    """
    if config.scf.method != "mixing":
        from .direct import run_direct_minimization

        if wavefunctions is not None or density is not None:
            raise NotImplementedError(
                f"scf.method: {config.scf.method} always starts from a fresh set of "
                "orbitals; restarting a direct minimisation from a stored density is not "
                "implemented. Use scf.method: mixing to restart."
            )
        return run_direct_minimization(system, config, logger=logger)

    log = logger or (lambda *_: None)
    started = time.perf_counter()
    grid = system.grid
    natoms = len(system.symbols)
    smearing = _smearing_in_hartree(config)

    density = system.initial_density() if density is None else np.array(density, dtype=float)
    wavefunctions = wavefunctions or _initial_wavefunctions(system, seed=12345)
    mixer = create_mixer(config.scf.mixing, grid, floating_charge=system.is_paw)
    if system.is_paw and isinstance(mixer.precondition, KerkerPreconditioner):
        # The occupancy matrices ride along in the same vector; only the density half sits
        # on a grid and can be Kerker-filtered. See BlockPreconditioner.
        mixer.precondition = BlockPreconditioner(mixer.precondition, density.shape)
    occupancies = _initial_occupancies(system) if system.is_paw else None

    energy_threshold = config.scf.tol_energy * natoms * EV_IN_HARTREE
    history: list[tuple[float, float, float]] = []
    previous_energy = np.inf
    residual = np.inf
    converged = False
    energies = Energies(ewald=system.ewald)
    occupation = None
    eigenvalues = np.zeros((system.nspin, system.nkpoints, system.nbands))
    iteration = 0

    log(
        f"  {'iter':>4s} {'E (eV)':>18s} {'dE (eV/atom)':>14s} "
        f"{'rho residual':>13s} {'eig sum res':>12s} {'time (s)':>9s}"
    )
    for iteration in range(1, config.scf.max_steps + 1):
        step_started = time.perf_counter()
        state = paw_state(system, occupancies) if system.is_paw else None
        potential, _, _, hartree_g = effective_potential(system, density, state)
        if state is not None:
            state.coupling = system.paw.coupling(state.onsite, hartree_g, system.nspin)
        hamiltonian = Hamiltonian(
            system, potential, coupling=None if state is None else state.coupling
        )

        # Loose diagonalisation early on is not a shortcut: the potential itself is still
        # wrong, so eigenvectors converged past that error are wasted work.
        eigen_tolerance = float(np.clip(1e-3 * min(residual, 1.0), 1e-11, 1e-6))
        max_eigen_steps = 40 if iteration == 1 else 20

        eigenvalues = np.empty((system.nspin, system.nkpoints, system.nbands))
        for spin in range(system.nspin):
            for ik, sphere in enumerate(system.wave_spheres):
                overlap = (
                    None
                    if not system.is_paw
                    else (lambda block, ik=ik, h=hamiltonian: h.apply_overlap(block, ik))
                )
                result = solve(
                    config.scf.eigensolver,
                    lambda block, ik=ik, spin=spin, h=hamiltonian: h.apply(block, ik, spin),
                    wavefunctions[spin][ik],
                    sphere.kinetic_energy,
                    tolerance=eigen_tolerance,
                    max_iterations=max_eigen_steps,
                    apply_overlap=overlap,
                )
                eigenvalues[spin, ik] = result.eigenvalues
                wavefunctions[spin][ik] = result.coefficients

        occupation = compute_occupations(
            eigenvalues,
            system.kpoints.weights,
            system.nelectrons,
            scheme=config.occupations.scheme,
            smearing=smearing,
            order=config.occupations.order,
            nspin=system.nspin,
        )
        if iteration == 1 and config.occupations.scheme == "fixed":
            _warn_about_fixed_occupations(occupation, log)
        density_out = compute_density(system, wavefunctions, occupation.occupations)
        occupancies_out = (
            compute_occupancies(system, wavefunctions, occupation.occupations)
            if system.is_paw
            else None
        )

        # The energy is the Kohn-Sham functional at the output density, so its Hartree and
        # exchange-correlation pieces are re-evaluated there rather than reused from n_in.
        state_out = paw_state(system, occupancies_out) if system.is_paw else None
        energy_hartree, energy_xc = _hartree_xc_energies(system, density_out, state_out)
        energies = Energies(
            kinetic=_kinetic_energy(system, wavefunctions, occupation.occupations),
            local=float(np.sum(system.v_local * _local_charge(system, density_out)))
            * grid.cell_volume_per_point,
            hartree=energy_hartree,
            xc=energy_xc,
            # In PAW the separable term is not an energy of its own: the coupling D_ij is
            # the derivative of the one-centre correction, and counting both would be a
            # double count. The correction itself is what goes here.
            nonlocal_=(
                state_out.energy
                if state_out is not None
                else _nonlocal_energy(system, wavefunctions, occupation.occupations)
            ),
            ewald=system.ewald,
            entropy=occupation.entropy_term,
        )

        band_sum = (
            2.0
            / system.nspin
            * float(
                np.einsum(
                    "k,skn,skn->", system.kpoints.weights, occupation.occupations, eigenvalues
                )
            )
        )
        separable = (
            float(np.einsum("sij,sij->", occupancies_out, state.coupling))
            if system.is_paw
            else energies.nonlocal_
        )
        identity = band_sum - (energies.kinetic + hamiltonian.local_energy(density_out) + separable)

        # The density residual is integrated |dn| per valence electron, which makes the
        # threshold mean the same thing for silicon and for iron; see SCFConfig.tol_density.
        difference = density_out - density
        residual = (
            float(np.sum(np.abs(difference))) * grid.cell_volume_per_point / system.nelectrons
        )
        energy = energies.free_energy
        delta = energy - previous_energy
        history.append((energy, residual, identity))
        log(
            f"  {iteration:4d} {energy * HARTREE_IN_EV:18.9f} "
            f"{delta * HARTREE_IN_EV / natoms:14.2e} {residual:13.3e} "
            f"{identity * HARTREE_IN_EV:12.2e} {time.perf_counter() - step_started:9.2f}"
        )

        if iteration > 1 and abs(delta) < energy_threshold and residual < config.scf.tol_density:
            converged = True
            break
        previous_energy = energy
        if system.is_paw:
            # One vector, one history. D_ij and the density are both functionals of the
            # same orbitals, so mixing them separately -- Pulay for one, a plain damped
            # step for the other -- makes the two describe different iterates and the
            # residual stops converging.
            packed = mixer(
                np.concatenate([density.ravel(), occupancies.ravel()]),
                np.concatenate([density_out.ravel(), occupancies_out.ravel()]),
            )
            split = density.size
            density = packed[:split].reshape(density.shape)
            occupancies = packed[split:].reshape(occupancies.shape)
        else:
            density = mixer(density, density_out)

    # The result carries the *output* density, converged or not: it is the one the
    # reported energies and the stored orbitals belong to, and the one a restart should
    # resume from. The mixer's next input is an extrapolation and belongs to no result.
    return SCFResult(
        converged=converged,
        iterations=iteration,
        energies=energies,
        eigenvalues=eigenvalues,
        occupations=occupation.occupations,
        fermi_level=occupation.fermi_level,
        gap=occupation.gap,
        magnetization=occupation.magnetization,
        density=density_out,
        wavefunctions=wavefunctions,
        history=history,
        electron_count=electron_count(system, density_out),
        seconds=time.perf_counter() - started,
    )


def _initial_occupancies(system: PlaneWaveSystem) -> np.ndarray:
    """The superposed-atom occupancy matrices that go with the starting density."""
    blocks = []
    for atom, symbol in enumerate(system.symbols):
        setup = system.paw.setups[symbol]
        moment = float(system.initial_magmoms[atom]) if system.nspin == 2 else 0.0
        blocks.append(setup.atomic_occupancies(system.nspin, moment))
    return system.paw.scatter(blocks, system.nspin)


def _smearing_in_hartree(config: Config) -> float:
    from ..units import parse_energy

    if config.occupations.scheme == "fixed":
        return 0.0
    return parse_energy(config.occupations.smearing, "eV")


def _hartree_xc_energies(
    system: PlaneWaveSystem, density: np.ndarray, state: PAWState | None = None
) -> tuple[float, float]:
    energy_hartree, _ = hartree(
        system.grid, system.density_sphere, _hartree_charge(system, density, state)
    )
    xc_density = density
    if system.core_density is not None:
        xc_density = density + system.core_density[None, ...] / system.nspin
    return energy_hartree, system.xc.evaluate_on_grid(xc_density, system.grid).energy


def compute_occupancies(
    system: PlaneWaveSystem, wavefunctions, occupations: np.ndarray
) -> np.ndarray:
    """``D^a_ij`` accumulated from the orbitals, shape ``(nspin, nproj, nproj)``.

    The same weights and spin degeneracy as the density, so that the two are consistent:
    an unpolarised calculation carries the total in its single channel.
    """
    degeneracy = 2.0 / system.nspin
    size = system.paw.n_projectors
    total = np.zeros((system.nspin, size, size))
    for spin in range(system.nspin):
        for ik in range(system.nkpoints):
            factors = degeneracy * system.kpoints.weights[ik] * occupations[spin, ik]
            total[spin] += system.projectors[ik].occupancies(wavefunctions[spin][ik], factors)
    # Only the on-site blocks are meaningful; projectors on different atoms are far apart
    # and their cross terms are not part of the PAW expansion.
    mask = np.zeros((size, size))
    for start, count in system.paw.blocks:
        mask[start : start + count, start : start + count] = 1.0
    return system.paw.symmetrize(total * mask)


def _kinetic_energy(system: PlaneWaveSystem, wavefunctions, occupations: np.ndarray) -> float:
    degeneracy = 2.0 / system.nspin
    total = 0.0
    for spin in range(system.nspin):
        for ik, sphere in enumerate(system.wave_spheres):
            factors = degeneracy * system.kpoints.weights[ik] * occupations[spin, ik]
            density = np.abs(wavefunctions[spin][ik]) ** 2
            total += float(factors @ (density @ sphere.kinetic_energy))
    return total


def _nonlocal_energy(system: PlaneWaveSystem, wavefunctions, occupations: np.ndarray) -> float:
    degeneracy = 2.0 / system.nspin
    total = 0.0
    for spin in range(system.nspin):
        for ik in range(system.nkpoints):
            factors = degeneracy * system.kpoints.weights[ik] * occupations[spin, ik]
            total += system.projectors[ik].energy(wavefunctions[spin][ik], factors)
    return total


def _warn_about_fixed_occupations(occupation, log) -> None:
    """Fixed filling needs a gap; without one the SCF flips between fillings and oscillates.

    ``occupations.scheme: fixed`` assigns each state 0 or 1 by energy order. If two states
    straddle the Fermi level within a few meV, a small change in the potential swaps them,
    the density jumps discontinuously, and the iteration cycles instead of converging --
    typically stalling with a density residual two or three orders above the threshold
    while the energy oscillates by tens of meV. Smearing removes the discontinuity.
    """
    gap = occupation.gap
    if gap is None:
        log(
            "  WARNING: occupations.scheme is 'fixed' but the states are not gapped at "
            "this k-point sampling. Fixed filling will oscillate; use a smearing scheme "
            "(gaussian, marzari-vanderbilt) instead"
        )
    elif gap * HARTREE_IN_EV < 0.05:
        log(
            f"  WARNING: the gap is only {gap * HARTREE_IN_EV:.4f} eV, which is too small "
            "for occupations.scheme: fixed to be stable. Use a smearing scheme"
        )
