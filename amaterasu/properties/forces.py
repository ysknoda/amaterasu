"""Forces on the ions: the derivative of the total energy with respect to the positions.

In a plane-wave basis the force is a pure Hellmann-Feynman quantity. The basis functions
are plane waves fixed to the cell, not orbitals attached to the atoms, so moving an atom
does not move the basis and there is **no Pulay force** -- the term that dominates the
bookkeeping in a localized-basis code simply does not exist here. (A Pulay *stress* does
exist, because straining the cell does change which plane waves fall inside the cutoff;
see :mod:`amaterasu.properties.stress`.)

What is left are the three energy terms that depend explicitly on where the atoms are:

``local pseudopotential``
    ``E_loc = Omega sum_G V_loc(G)* n(G)`` with ``V_loc(G)`` carrying ``exp(-i G.tau_a)``.
``nonlocal (Kleinman-Bylander)``
    the projectors carry ``exp(-i (k+G).tau_a)``, so each ``<beta|psi>`` moves with its atom.
``nonlinear core correction``
    the frozen partial core density is centred on the atom and enters ``E_xc``.

plus the Ewald sum, which is handled in :mod:`amaterasu.hamiltonian.ewald`.

Symmetrisation is not cosmetic here. The Brillouin-zone sum runs over the irreducible
wedge, and while that is enough for a scalar, a *vector* accumulated over the wedge is not
the full-zone vector: the wedge sum has to be averaged over the point group to recover it.
Skipping the average leaves forces that break the crystal's symmetry -- an atom on a
high-symmetry site acquires a force it cannot physically have.
"""

from __future__ import annotations

import numpy as np

from ..hamiltonian.ewald import ewald
from ..hamiltonian.local import structure_factors
from ..hamiltonian.system import PlaneWaveSystem
from ..pseudo.formfactors import density_form_factor, local_potential_form_factor, radial_transform
from ..system.symmetry import symmetrize_vectors

__all__ = ["ForceContributions", "compute_forces"]


class ForceContributions(dict):
    """The force decomposition, in Ha/Bohr. Summing the values gives the total."""

    @property
    def total(self) -> np.ndarray:
        return sum(self.values())


def compute_forces(
    system: PlaneWaveSystem,
    density: np.ndarray,
    wavefunctions,
    occupations: np.ndarray,
    *,
    eigenvalues: np.ndarray | None = None,
    symmetrize: bool = True,
    decompose: bool = False,
) -> np.ndarray | ForceContributions:
    """Forces on every atom in Ha/Bohr, shape ``(natoms, 3)``.

    ``decompose=True`` returns the individual contributions instead of their sum, which is
    what the finite-difference tests check term by term. ``eigenvalues`` is required for a
    PAW calculation and ignored otherwise: PAW's overlap operator moves with the atoms, so
    the orthonormality constraint contributes a term weighted by the eigenvalues -- the
    same structure as a Pulay force, except that here it is exact rather than a correction.
    """
    grid = system.grid
    sphere = system.density_sphere
    gvectors = sphere.cartesian
    phases = structure_factors(sphere, system.positions)  # exp(-i G.tau_a)

    total_density = density.sum(axis=0)
    density_g = grid.gather(grid.to_reciprocal(total_density.astype(complex)), sphere)

    contributions = ForceContributions()
    if system.is_paw:
        contributions.update(_paw_forces(system, density, wavefunctions, occupations, eigenvalues))
        if symmetrize and system.symmetry is not None:
            for name, value in contributions.items():
                contributions[name] = symmetrize_vectors(
                    value, system.atoms, system.symmetry, system.cell
                )
        return contributions if decompose else contributions.total

    charges = np.array([system.pseudopotentials[s].valence_charge for s in system.symbols])
    contributions["ewald"] = ewald(system.cell, system.positions, charges).forces
    contributions["local"] = _local_forces(system, sphere, gvectors, phases, density_g)
    contributions["nonlocal"] = _nonlocal_forces(system, wavefunctions, occupations)

    if system.core_density is not None:
        # The frozen core enters only E_xc, and it is shared equally between the spin
        # channels, so the potential conjugate to it is the channel average of v_xc.
        # Evaluating the functional directly is cheaper and better conditioned than
        # subtracting the local and Hartree parts back out of v_eff.
        xc_density = density + system.core_density[None, ...] / system.nspin
        xc = system.xc.evaluate_on_grid(xc_density, grid)
        contributions["core"] = _core_forces(
            system, sphere, gvectors, phases, xc.potential.mean(axis=0)
        )

    if symmetrize and system.symmetry is not None:
        for name, value in contributions.items():
            contributions[name] = symmetrize_vectors(
                value, system.atoms, system.symmetry, system.cell
            )
    return contributions if decompose else contributions.total


def _local_forces(system, sphere, gvectors, phases, density_g) -> np.ndarray:
    """``F_a = -sum_G i G v_a(|G|) exp(i G.tau_a) n(G)``.

    Each ``+G``/``-G`` pair contributes a conjugate pair, so the sum is real; the
    imaginary part left over is a check on the sphere-to-grid bookkeeping rather than a
    physical quantity, and is asserted to be negligible.
    """
    q = np.sqrt(sphere.g2)
    form_factors = {
        symbol: local_potential_form_factor(pseudo, q)
        for symbol, pseudo in system.pseudopotentials.items()
    }
    forces = np.zeros((len(system.symbols), 3))
    for atom, symbol in enumerate(system.symbols):
        weight = 1j * form_factors[symbol] * phases[atom].conj() * density_g
        forces[atom] = -_real(np.einsum("g,gi->i", weight, gvectors), "local force")
    return forces


def _core_forces(system, sphere, gvectors, phases, xc_potential) -> np.ndarray:
    """Force from the partial core density moving with its atom (the NLCC force)."""
    q = np.sqrt(sphere.g2)
    potential_g = system.grid.gather(
        system.grid.to_reciprocal(xc_potential.astype(complex)), sphere
    )
    forces = np.zeros((len(system.symbols), 3))
    for atom, symbol in enumerate(system.symbols):
        pseudo = system.pseudopotentials[symbol]
        if pseudo.core_density is None:
            continue
        core_g = density_form_factor(pseudo.grid, pseudo.core_density, q)
        weight = 1j * potential_g.conj() * core_g * phases[atom]
        forces[atom] = _real(np.einsum("g,gi->i", weight, gvectors), "core force")
    return forces


def _nonlocal_forces(system, wavefunctions, occupations) -> np.ndarray:
    """Derivative of the Kleinman-Bylander energy through the projector phases."""
    degeneracy = 2.0 / system.nspin
    forces = np.zeros((len(system.symbols), 3))
    atom_of_row = None
    for spin in range(system.nspin):
        for ik, sphere in enumerate(system.wave_spheres):
            projectors = system.projectors[ik]
            if len(projectors) == 0:
                continue
            if atom_of_row is None:
                atom_of_row = np.array([label[0] for label in projectors.labels])
            block = wavefunctions[spin][ik]
            weights = degeneracy * system.kpoints.weights[ik] * occupations[spin, ik]
            active = weights > 1e-12
            if not np.any(active):
                continue
            block, weights = block[active], weights[active]

            projected = projectors.project(block)  # (nbands, nproj)
            coupled = projected @ projectors.dij
            for axis in range(3):
                # d<beta_j|psi>/d tau_a = sum_G i(k+G) conj(P_j(G)) c_G
                derivative = (block * (1j * sphere.cartesian[:, axis])) @ projectors.matrix.conj().T
                terms = 2.0 * np.real(coupled.conj() * derivative)
                per_projector = np.einsum("n,nj->j", weights, terms)
                np.add.at(forces[:, axis], atom_of_row, -per_projector)
    return forces


def _real(vector: np.ndarray, what: str) -> np.ndarray:
    residual = float(np.max(np.abs(vector.imag)))
    scale = max(float(np.max(np.abs(vector.real))), 1e-30)
    if residual > 1e-6 * scale and residual > 1e-12:
        raise ValueError(
            f"the {what} came out complex (imaginary part {residual:.3e} against "
            f"{scale:.3e}); a phase convention or the sphere-to-grid map is wrong"
        )
    return vector.real


# --------------------------------------------------------------------------------- PAW
# A PAW force has four pieces, and none of them is the local pseudopotential term of a
# norm-conserving run, because there is no local ionic pseudopotential. What moves with an
# atom is: its smooth core density, its zero potential, its compensation charge, and its
# projectors. The last of these carries two contributions -- the energy's dependence on
# the occupancies, and the orthonormality constraint, because the PAW overlap operator
# moves with the atoms too. That second piece is the exact analogue of a Pulay force, and
# unlike a Pulay force it is not an approximation to be minimised but a term to be
# computed; leaving it out is a silent error of the same size as the force itself.


def _paw_forces(system, density, wavefunctions, occupations, eigenvalues):
    """The four position-dependent terms of the PAW energy."""
    from ..scf.driver import compute_occupancies, effective_potential, paw_state

    if eigenvalues is None:
        raise ValueError(
            "PAW forces need the eigenvalues: the overlap operator moves with the atoms, "
            "so the orthonormality constraint contributes a term weighted by them"
        )
    grid, sphere = system.grid, system.density_sphere
    gvectors = sphere.cartesian
    paw = system.paw

    occupancies = compute_occupancies(system, wavefunctions, occupations)
    state = paw_state(system, occupancies)
    potential, _, _, hartree_g = effective_potential(system, density, state)
    coupling = paw.coupling(state.onsite, hartree_g, system.nspin)

    contributions = ForceContributions()
    phases = structure_factors(sphere, system.positions)
    q = np.sqrt(sphere.g2)

    # The smooth core is a charge, an exchange-correlation density and a partner of the
    # zero potential all at once, so the potential conjugate to it is the whole of v_eff,
    # averaged over spin because the core is not polarised.
    mean_potential = potential.mean(axis=0)
    mean_g = grid.gather(grid.to_reciprocal(mean_potential.astype(complex)), sphere)
    core = np.zeros((len(system.symbols), 3))
    for atom, symbol in enumerate(system.symbols):
        data = paw.setups[symbol].dataset
        form = density_form_factor(data.grid, data.core_density, q)
        weight = 1j * mean_g.conj() * form * phases[atom]
        core[atom] = _real(np.einsum("g,gi->i", weight, gvectors), "PAW core force")
    contributions["core"] = core

    # The zero potential couples to n~, which includes the smooth core.
    smooth = density.sum(axis=0)
    if system.core_density is not None:
        smooth = smooth + system.core_density
    smooth_g = grid.gather(grid.to_reciprocal(smooth.astype(complex)), sphere)
    vbar = np.zeros_like(core)
    for atom, symbol in enumerate(system.symbols):
        setup = paw.setups[symbol]
        data = setup.dataset
        values = np.zeros(len(data.grid))
        values[: setup.stop] = (
            np.sqrt(4.0 * np.pi)
            * data.zero_potential[: setup.stop]
            * data.grid.r[: setup.stop] ** 2
        )
        form = radial_transform(data.grid, values, 0, q)
        weight = 1j * form * phases[atom].conj() * smooth_g
        vbar[atom] = -_real(np.einsum("g,gi->i", weight, gvectors), "PAW zero-potential force")
    contributions["vbar"] = vbar

    # The compensation charge rides on its atom; its multipoles do not move with it, so
    # only the explicit phase differentiates here. The response *through* the multipoles
    # is part of the projector term, via H^a_ij.
    gradients = paw.shape_projection_gradients(hartree_g, gvectors)
    compensation = np.array(
        [
            -np.einsum("L,Lc->c", result.multipoles, gradient)
            for result, gradient in zip(state.onsite, gradients, strict=True)
        ]
    )
    contributions["compensation"] = compensation

    contributions["projectors"] = _paw_projector_forces(
        system, wavefunctions, occupations, eigenvalues, coupling
    )
    return contributions


def _paw_projector_forces(system, wavefunctions, occupations, eigenvalues, coupling):
    """``-sum_n f_n sum_ij (H^a_ij - eps_n O^a_ij) d/dR <psi|p_i><p_j|psi>``, Eq. (50)."""
    degeneracy = 2.0 / system.nspin
    forces = np.zeros((len(system.symbols), 3))
    overlap = system.paw.overlap
    atom_of_row = np.concatenate(
        [np.full(size, atom) for atom, (_, size) in enumerate(system.paw.blocks)]
    )
    for spin in range(system.nspin):
        for ik, sphere in enumerate(system.wave_spheres):
            projectors = system.projectors[ik]
            if len(projectors) == 0:  # pragma: no cover - a cell with no projectors
                continue
            weights = degeneracy * system.kpoints.weights[ik] * occupations[spin, ik]
            active = weights > 1e-12
            if not np.any(active):
                continue
            block = wavefunctions[spin][ik][active]
            levels = eigenvalues[spin, ik][active]
            weights = weights[active]

            projected = projectors.project(block)
            # (H - eps_n O) applied on the right, band by band.
            coupled = projected @ coupling[spin] - levels[:, None] * (projected @ overlap)
            for axis in range(3):
                derivative = (block * (1j * sphere.cartesian[:, axis])) @ projectors.matrix.conj().T
                terms = 2.0 * np.real(coupled.conj() * derivative)
                np.add.at(forces[:, axis], atom_of_row, -np.einsum("n,nj->j", weights, terms))
    return forces
