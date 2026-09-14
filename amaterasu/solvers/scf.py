"""The ``solver: scf`` entry point: a single self-consistent field calculation.

This is the thin layer between a validated input file and
:func:`amaterasu.scf.driver.run_scf`. It resolves the pseudopotentials, fixes the
cutoffs, builds the :class:`~amaterasu.hamiltonian.system.PlaneWaveSystem`, runs the
loop, prints a report in the output units and writes the HDF5 result.

Every later solver -- relaxation, molecular dynamics, band structures -- drives this same
machinery, so the split is deliberate: the physics lives in ``scf/``, and this module only
handles input, reporting and output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from ..config.schema import Config
from ..hamiltonian.system import PlaneWaveSystem, build_system
from ..io.results import write_scf
from ..pseudo.registry import load_pseudopotentials, resolve_cutoffs
from ..scf.driver import SCFResult, run_scf
from ..units import HARTREE_IN_EV

__all__ = ["SCFRun", "prepare", "report", "run"]


@dataclass
class SCFRun:
    """A completed SCF calculation and the system it was run on."""

    system: PlaneWaveSystem
    result: SCFResult
    pseudopotential_paths: dict[str, str]


def prepare(
    atoms: Atoms, config: Config, *, download: bool = True, logger=print
) -> tuple[PlaneWaveSystem, dict[str, str]]:
    """Resolve datasets and cutoffs and build the plane-wave system."""
    symbols = sorted(set(atoms.get_chemical_symbols()))
    resolved = load_pseudopotentials(config, symbols, download=download)
    for warning in resolved.warnings:
        logger(f"  WARNING: {warning}")
    ecut, ecut_rho = resolve_cutoffs(config, resolved)
    if resolved.recommended_ecut is not None and ecut < resolved.recommended_ecut:
        logger(
            f"  WARNING: basis.ecut = {ecut:g} Ha is below the dataset's recommended "
            f"{resolved.recommended_ecut:g} Ha; the result may be under-converged"
        )
    system = build_system(atoms, config, resolved.pseudopotentials, ecut, ecut_rho)
    for warning in system.warnings:
        logger(f"  WARNING: {warning}")
    paths = {
        symbol: str(getattr(pseudo, "source", "") or "")
        for symbol, pseudo in resolved.pseudopotentials.items()
    }
    return system, paths


def report(system: PlaneWaveSystem, result: SCFResult) -> str:
    """The human-readable summary printed after the loop, in the output units."""
    natoms = len(system.symbols)
    lines = [
        "",
        f"SCF {'converged' if result.converged else 'DID NOT CONVERGE'} in "
        f"{result.iterations} iterations ({result.seconds:.1f} s)",
        result.energies.render(natoms),
        "",
        f"  {'electrons in the cell':<26s} {result.electron_count:16.9f}",
        f"  {'Fermi level':<26s} {result.fermi_level * HARTREE_IN_EV:16.9f} eV",
    ]
    if result.gap is not None:
        edges = result.homo_lumo
        lines.append(f"  {'band gap':<26s} {result.gap * HARTREE_IN_EV:16.9f} eV")
        if edges is not None:
            lines.append(
                f"  {'  highest occupied':<26s} {edges[0] * HARTREE_IN_EV:16.9f} eV"
                f"   lowest unoccupied {edges[1] * HARTREE_IN_EV:.9f} eV"
            )
    else:
        lines.append(f"  {'band gap':<26s} {'none (metallic occupations)':>16s}")
    if system.nspin == 2:
        lines.append(
            f"  {'magnetic moment':<26s} {result.magnetization:16.6f} muB "
            f"({result.magnetization / natoms:.6f} per atom)"
        )
    if result.history:
        lines.append(
            f"  {'eigenvalue-sum identity':<26s} "
            f"{result.history[-1][2] * HARTREE_IN_EV:16.3e} eV   "
            "(sum w f eps - [T + int v_eff n + E_nl]; zero by construction)"
        )
    lines.append("")
    lines.append(_eigenvalue_table(system, result))
    return "\n".join(lines)


def _eigenvalue_table(system: PlaneWaveSystem, result: SCFResult, limit: int = 6) -> str:
    """Eigenvalues at the first few k-points, relative to the Fermi level."""
    lines = ["  eigenvalues (eV, relative to the Fermi level)"]
    fermi = result.fermi_level
    for ik in range(min(limit, system.nkpoints)):
        coordinates = " ".join(f"{x:6.3f}" for x in system.kpoints.points[ik])
        for spin in range(system.nspin):
            label = f"    k=({coordinates}) w={system.kpoints.weights[ik]:.4f}"
            if system.nspin == 2:
                label += f" {'up' if spin == 0 else 'down'}"
            values = (result.eigenvalues[spin, ik] - fermi) * HARTREE_IN_EV
            occupations = result.occupations[spin, ik]
            lines.append(label)
            lines.append(
                "      "
                + "  ".join(f"{e:8.3f}({f:4.2f})" for e, f in zip(values, occupations, strict=True))
            )
    if system.nkpoints > limit:
        lines.append(f"    ... and {system.nkpoints - limit} more k-points")
    return "\n".join(lines)


def _restart_state(system: PlaneWaveSystem, path: Path, logger):
    """Load a density (and orbitals, if stored) from a previous run, if they still fit.

    A checkpoint is only usable with the same grid and the same k-point set; anything
    else -- a different cutoff, a different mesh, a moved atom -- silently means a
    different basis. Rather than interpolate, the mismatch is reported and the run starts
    from the atomic superposition, which costs a few extra iterations and no correctness.
    """
    from ..io.results import read_scf

    stored = read_scf(path)
    density = np.asarray(stored["density"], dtype=float)
    if density.shape != (system.nspin, *system.grid.shape):
        logger(
            f"  NOTE: {path} holds a density of shape {density.shape}, but this run needs "
            f"{(system.nspin, *system.grid.shape)}; starting from atomic densities instead"
        )
        return None, None
    logger(f"  restarting from the density in {path}")
    wavefunctions = stored.get("wavefunctions")
    if wavefunctions is not None:
        shapes_match = len(wavefunctions) == system.nspin and all(
            len(per_spin) == system.nkpoints
            and all(
                block.shape == (system.nbands, len(sphere))
                for block, sphere in zip(per_spin, system.wave_spheres, strict=True)
            )
            for per_spin in wavefunctions
        )
        if not shapes_match:
            logger(f"  NOTE: the orbitals in {path} do not match this basis; ignoring them")
            wavefunctions = None
        else:
            logger("  and from the stored orbitals")
    return density, wavefunctions


def run(
    atoms: Atoms,
    config: Config,
    *,
    output_directory: Path | None = None,
    logger=print,
    download: bool = True,
    write_output: bool = True,
    restart: str | Path | None = None,
) -> SCFRun:
    """Run one SCF calculation and write its result."""
    system, paths = prepare(atoms, config, download=download, logger=logger)
    logger(str(system))
    for warning in _accuracy_notes(system, config):
        logger(f"  NOTE: {warning}")

    density, wavefunctions = (None, None)
    if restart is not None:
        density, wavefunctions = _restart_state(system, Path(restart).expanduser(), logger)
    logger("")

    result = run_scf(system, config, logger=logger, density=density, wavefunctions=wavefunctions)
    logger(report(system, result))

    if write_output:
        directory = Path(output_directory or config.output.directory).expanduser()
        target = directory / f"{config.output.prefix}.h5"
        write_scf(
            target,
            system,
            result,
            config=config,
            pseudopotential_paths={s: p for s, p in paths.items() if p},
            save_wavefunctions=config.output.save_wavefunctions,
        )
        logger(f"\nwrote {target}")
    return SCFRun(system=system, result=result, pseudopotential_paths=paths)


def _accuracy_notes(system: PlaneWaveSystem, config: Config) -> list[str]:
    """Warnings that depend on the assembled system rather than the raw input."""
    notes = []
    if config.occupations.scheme == "fixed" and system.nspin == 1:
        notes.append(
            "occupations.scheme is 'fixed': the bands are filled in energy order, which "
            "is right for an insulator and unstable for a metal"
        )
    if len(system.kpoints) == 1 and np.allclose(system.kpoints.points[0], 0.0):
        notes.append(
            "only the Gamma point is sampled; this converges very slowly for anything "
            "but a large cell"
        )
    if system.symmetry is None:
        notes.append("running without symmetrisation; the density is not averaged over the group")
    if system.nspin == 2 and not np.any(np.abs(system.initial_magmoms) > 1e-8):
        notes.append(
            "system.spin is 'polarized' but every initial moment is zero. The "
            "non-magnetic solution is a fixed point of the spin-polarised equations, so "
            "the run will simply reproduce it at twice the cost. Set magmom= on the atoms "
            "in the structure file to look for a magnetic solution"
        )
    return notes
