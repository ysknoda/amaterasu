"""The ``solver: bands`` entry point: a band structure along a path in the Brillouin zone.

A band structure is not a self-consistent calculation. The density is converged once on a
regular mesh, and then *frozen*: the Kohn-Sham Hamiltonian it defines is diagonalised at
each k-point of the path, which are far too few and far too unevenly distributed to
integrate anything. Two consequences follow, and both are easy to get wrong.

**The Fermi level comes from the mesh, not from the path.** Occupations along a path are
meaningless -- the path is a one-dimensional slice, not a sample of the zone -- so the
reference energy is the one the self-consistent run found, and it is carried over
unchanged. Reporting a gap or a Fermi level computed from path eigenvalues would be
reporting a number about a set of points nobody chose for that purpose.

**The path is defined in a particular cell.** ``bands.path: auto`` asks seekpath for the
standard path of Hinuma et al., Comput. Mater. Sci. 128, 140 (2017), which is defined in
seekpath's own standardised primitive cell. That cell need not be the one in the input
file, and if it is not, the labels ``X``, ``L``, ``W`` name points of a different Brillouin
zone than the one the input describes. This module therefore runs the whole calculation --
self-consistency included -- in the cell the path came back in, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from ..config.schema import Config
from ..hamiltonian.operator import Hamiltonian
from ..hamiltonian.system import PlaneWaveSystem, build_system
from ..scf.driver import compute_occupancies, effective_potential, paw_state
from ..scf.eigen import random_bands, solve
from ..system.kpoints import band_path_with_cell
from ..units import HARTREE_IN_EV
from . import scf as scf_solver

__all__ = ["BandStructure", "run", "solve_along_path"]


@dataclass
class BandStructure:
    """Eigenvalues along a path, with the reference energy the SCF run established."""

    #: The :class:`ase.dft.kpoints.BandPath` the eigenvalues were computed on.
    path: object
    #: Shape ``(nspin, nkpoints, nbands)`` in Hartree.
    eigenvalues: np.ndarray
    #: Fermi level in Hartree, from the self-consistent mesh.
    fermi_level: float
    #: The self-consistent run the density came from.
    scf: scf_solver.SCFRun
    #: True when the path forced a different cell than the input structure.
    restandardised: bool = False

    @property
    def gap(self) -> float | None:
        """Direct-plus-indirect gap over the sampled points, in Hartree, or ``None``.

        A path visits far fewer points than a mesh, so this is a lower bound on the true
        indirect gap: the band extremum may lie between two sampled points, or off the
        path entirely. It is quoted because it is what the picture shows, not as a
        converged property.
        """
        below = self.eigenvalues[self.eigenvalues <= self.fermi_level]
        above = self.eigenvalues[self.eigenvalues > self.fermi_level]
        if below.size == 0 or above.size == 0:
            return None
        gap = float(above.min() - below.max())
        return gap if gap > 0 else None

    def render(self) -> str:
        lines = [f"band structure along {self.path.path} ({len(self.eigenvalues[0])} k-points)"]
        if self.restandardised:
            lines.append(
                "  the path is seekpath's, so the calculation was run in seekpath's "
                "standardised primitive cell rather than the input cell"
            )
        fermi = self.fermi_level * HARTREE_IN_EV
        lines.append(f"  Fermi level (from the self-consistent mesh)  {fermi:12.6f} eV")
        gap = self.gap
        if gap is None:
            lines.append("  no gap along the path: the bands cross the Fermi level")
        else:
            lines.append(
                f"  smallest gap seen on the path                {gap * HARTREE_IN_EV:12.6f} eV"
                "  (a lower bound: the extremum may be off the path)"
            )
        return "\n".join(lines)


def solve_along_path(
    scf_run: scf_solver.SCFRun, config: Config, kpoints: np.ndarray, *, logger=print
) -> np.ndarray:
    """Diagonalise the converged Hamiltonian at ``kpoints``; ``(nspin, nk, nbands)`` in Hartree.

    The system is rebuilt for the new k-points because the plane-wave sphere, the
    projectors and the overlap blocks are all k-dependent. Everything that is not --
    the FFT grid, the local potential, the density -- is reused exactly, so the
    Hamiltonian being diagonalised here is the same operator the SCF converged.
    """
    system = scf_run.system
    result = scf_run.result

    band_config = _explicit_kpoint_config(config, kpoints)
    bands = build_system(
        system.atoms, band_config, system.pseudopotentials, system.ecut, system.ecut_rho
    )
    _check_reused_potential(system, bands)

    state = None
    if bands.is_paw:
        occupancies = compute_occupancies(system, result.wavefunctions, result.occupations)
        state = paw_state(bands, occupancies)
    potential, _, _, hartree_g = effective_potential(bands, result.density, state)
    coupling = None if state is None else bands.paw.coupling(state.onsite, hartree_g, bands.nspin)
    hamiltonian = Hamiltonian(bands, potential, coupling)

    eigenvalues = np.empty((bands.nspin, len(bands.kpoints), bands.nbands))
    for spin in range(bands.nspin):
        for ik, sphere in enumerate(bands.wave_spheres):
            guess = random_bands(
                len(sphere), bands.nbands, sphere.kinetic_energy, seed=7919 + 31 * spin + ik
            )
            overlap = (
                None
                if not bands.is_paw
                else (lambda block, ik=ik, h=hamiltonian: h.apply_overlap(block, ik))
            )
            solution = solve(
                config.scf.eigensolver,
                lambda block, ik=ik, spin=spin, h=hamiltonian: h.apply(block, ik, spin),
                guess,
                sphere.kinetic_energy,
                tolerance=config.scf.tol_energy / max(bands.nbands, 1),
                max_iterations=200,
                apply_overlap=overlap,
            )
            eigenvalues[spin, ik] = solution.eigenvalues
        logger(f"  spin {spin + 1}/{bands.nspin}: {len(bands.wave_spheres)} k-points diagonalised")
    return eigenvalues


def _explicit_kpoint_config(config: Config, kpoints: np.ndarray) -> Config:
    """A copy of ``config`` whose k-points are exactly ``kpoints``, unweighted and unreduced.

    Symmetry reduction is switched off deliberately: a path is chosen point by point, and
    folding it into the irreducible wedge would silently drop the points that make the
    picture.
    """
    from copy import deepcopy

    band_config = deepcopy(config)
    band_config.kpoints.scheme = "explicit"
    band_config.kpoints.points = [[float(x) for x in point] for point in np.asarray(kpoints)]
    band_config.kpoints.weights = None
    band_config.kpoints.symmetry = False
    return band_config


def _check_reused_potential(scf_system: PlaneWaveSystem, bands: PlaneWaveSystem) -> None:
    """The frozen density only means anything if the two systems share a grid."""
    if scf_system.grid.shape != bands.grid.shape:
        raise ValueError(  # pragma: no cover - defensive
            f"the band-structure system has FFT grid {bands.grid.shape} but the "
            f"self-consistent one had {scf_system.grid.shape}; the density cannot be reused"
        )


def run(
    atoms: Atoms,
    config: Config,
    *,
    output_directory: Path | None = None,
    logger=print,
    download: bool = True,
    write_output: bool = True,
) -> BandStructure:
    """Converge a density on the k-mesh, then diagonalise along the band path."""
    path, standard = band_path_with_cell(
        atoms, config.bands.path, config.bands.npoints, config.kpoints.symprec
    )
    restandardised = standard is not atoms and not _same_cell(atoms, standard)
    if restandardised:
        logger(
            "  NOTE: bands.path is 'auto', so the path comes from seekpath's standardised "
            "primitive cell; the calculation is run in that cell so the labels mean what "
            "they say"
        )
        standard.set_initial_magnetic_moments(_transferred_magmoms(atoms, standard))
        atoms = standard

    logger(f"\nself-consistent density on the {config.kpoints.scheme} mesh")
    scf_run = scf_solver.run(
        atoms,
        config,
        output_directory=output_directory,
        logger=logger,
        download=download,
        write_output=False,
        compute_properties=False,
    )
    if not scf_run.result.converged:
        raise RuntimeError("the self-consistent step did not converge; the bands would be noise")

    logger(f"\nnon-self-consistent diagonalisation along {path.path}")
    eigenvalues = solve_along_path(scf_run, config, path.kpts, logger=logger)
    structure = BandStructure(
        path=path,
        eigenvalues=eigenvalues,
        fermi_level=scf_run.result.fermi_level,
        scf=scf_run,
        restandardised=restandardised,
    )
    logger("\n" + structure.render())

    if write_output:
        from ..io.results import write_bands

        directory = Path(output_directory or config.output.directory).expanduser()
        target = directory / f"{config.output.prefix}_bands.h5"
        write_bands(target, structure, config=config)
        logger(f"\nwrote {target}")
    return structure


def _same_cell(atoms: Atoms, other: Atoms, tolerance: float = 1e-8) -> bool:
    return len(atoms) == len(other) and bool(
        np.allclose(np.asarray(atoms.cell), np.asarray(other.cell), atol=tolerance)
    )


def _transferred_magmoms(atoms: Atoms, standard: Atoms) -> np.ndarray:
    """Carry initial moments across the restandardisation, by species.

    Site-by-site transfer would need the mapping seekpath does not return; the moments in
    an input file are a starting guess per element in every case seen so far, and a guess
    that is right per element is enough to break the symmetry the SCF needs broken.
    """
    moments = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    if not np.any(np.abs(moments) > 1e-12):
        return np.zeros(len(standard))
    by_symbol: dict[str, float] = {}
    for symbol, moment in zip(atoms.get_chemical_symbols(), moments, strict=True):
        by_symbol.setdefault(symbol, float(moment))
    return np.array([by_symbol.get(s, 0.0) for s in standard.get_chemical_symbols()])
