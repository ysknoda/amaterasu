"""Geometry optimisation, driven through ``ase.optimize`` and ``ase.filters``.

None of the optimisation algorithms are reimplemented here. ASE's FIRE, L-BFGS, BFGS and
ODE12r are well tested, widely used and completely independent of the physics; what
amaterasu supplies is the calculator they call. This module is the adapter: it maps the
YAML settings onto ASE objects, attaches the constraints, runs the optimiser and reports.

Cell relaxation
---------------
A variable-cell run wraps the atoms in a filter that exposes the six strain degrees of
freedom alongside the positions, so a single optimiser handles both. ``FrechetCellFilter``
is the default because it parametrises the cell by the *matrix logarithm* of the
deformation, which makes the optimiser's implicit assumption -- that the energy is roughly
quadratic in its variables -- hold far better under large volume changes than the linear
strain of ``UnitCellFilter`` does.

And the warning that has to accompany it: the stress carries a Pulay contribution at any
finite cutoff (see :mod:`amaterasu.properties.stress`). A variable-cell relaxation
minimises the *computed* stress, so it settles at the volume where the physical and the
basis-set pressures cancel, not where the physical one vanishes. The remedy is to converge
the cutoff, or to relax again at a fixed cell once the volume has stopped moving.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.filters import ExpCellFilter, FrechetCellFilter, StrainFilter, UnitCellFilter
from ase.optimize import BFGS, FIRE, FIRE2, LBFGS, LBFGSLineSearch, ODE12r

from ..calculator import Amaterasu
from ..config.schema import Config
from ..io.structure_write import write_amaterasu_structure
from ..system.symmetry import analyze
from ..units import GPA_IN_EV_PER_ANG3

__all__ = ["RelaxResult", "run"]

_OPTIMIZERS = {
    "fire": FIRE,
    "fire2": FIRE2,
    "lbfgs": LBFGS,
    "lbfgs-ls": LBFGSLineSearch,
    "bfgs": BFGS,
    "ode12r": ODE12r,
}

_FILTERS = {
    "frechet": FrechetCellFilter,
    "unitcell": UnitCellFilter,
    "exp": ExpCellFilter,
    "strain": StrainFilter,
}


@dataclass
class RelaxResult:
    """What a relaxation produced."""

    atoms: Atoms
    converged: bool
    steps: int
    #: Maximum force in eV/Angstrom at the final geometry.
    fmax: float
    initial_energy: float
    final_energy: float
    #: Space group number before and after, at the input ``symprec``.
    spacegroup: tuple[int, int]
    trajectory: Path | None = None

    def render(self) -> str:
        change = self.final_energy - self.initial_energy
        lines = [
            "",
            f"relaxation {'converged' if self.converged else 'DID NOT CONVERGE'} "
            f"in {self.steps} steps",
            f"  {'maximum force':<26s} {self.fmax:16.9f} eV/Angstrom",
            f"  {'energy change':<26s} {change:16.9f} eV",
            f"  {'final energy':<26s} {self.final_energy:16.9f} eV",
            f"  {'space group':<26s} {self.spacegroup[0]} -> {self.spacegroup[1]}",
        ]
        if self.spacegroup[0] != self.spacegroup[1]:
            lines.append(
                "  NOTE: the space group changed during the relaxation. Rising is the "
                "usual case -- a distorted start relaxing back onto its ideal sites -- "
                "and falling means a symmetry-breaking instability was found. Either can "
                "be right; relax.fix_symmetry: true forbids both."
            )
        cell = self.atoms.cell.cellpar()
        lines.append(
            f"  {'final cell':<26s} a={cell[0]:.6f} b={cell[1]:.6f} c={cell[2]:.6f} A, "
            f"alpha={cell[3]:.4f} beta={cell[4]:.4f} gamma={cell[5]:.4f} deg"
        )
        lines.append(f"  {'volume':<26s} {self.atoms.get_volume():16.9f} Angstrom^3")
        return "\n".join(lines)


def run(
    atoms: Atoms,
    config: Config,
    *,
    output_directory: Path | None = None,
    logger=print,
    download: bool = True,
) -> RelaxResult:
    """Relax ``atoms`` according to ``config.relax``."""
    settings = config.relax
    optimizer_class = _OPTIMIZERS.get(settings.algorithm)
    if optimizer_class is None:  # pragma: no cover - the schema restricts the value
        raise ValueError(f"unknown relax.algorithm {settings.algorithm!r}")

    working = atoms.copy()
    working.calc = Amaterasu(config, logger=logger, download=download)
    symprec = config.kpoints.symprec
    initial_group = _spacegroup(working, symprec)

    if settings.fix_symmetry:
        from ase.constraints import FixSymmetry

        working.set_constraint([*working.constraints, FixSymmetry(working, symprec=symprec)])
        logger(f"  constraining the space group at {initial_group} (relax.fix_symmetry)")

    initial_energy = working.get_potential_energy()

    target = working
    if settings.cell:
        filter_class = _FILTERS[settings.cell_filter]
        if filter_class is StrainFilter:
            # StrainFilter varies only the cell, and takes no external pressure.
            if settings.pressure:
                logger(
                    f"  WARNING: relax.cell_filter 'strain' cannot apply an external "
                    f"pressure; ignoring relax.pressure = {settings.pressure:g} GPa. Use "
                    "'frechet' for a target pressure"
                )
            target = filter_class(working)
        else:
            target = filter_class(working, scalar_pressure=settings.pressure * GPA_IN_EV_PER_ANG3)
        logger(_pulay_warning(settings.pressure))

    directory = Path(output_directory or config.output.directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    trajectory = directory / f"{config.output.prefix}_relax.traj"
    optimizer = optimizer_class(target, logfile=None, trajectory=str(trajectory))

    step = {"n": 0}

    def report():
        step["n"] += 1
        forces = working.get_forces()
        logger(
            f"  step {step['n']:4d}  E = {working.get_potential_energy():18.9f} eV   "
            f"fmax = {np.sqrt((forces**2).sum(axis=1)).max():10.6f} eV/A   "
            f"V = {working.get_volume():12.6f} A^3"
        )

    optimizer.attach(report, interval=1)
    converged = optimizer.run(fmax=settings.fmax, steps=settings.max_steps)

    forces = working.get_forces()
    result = RelaxResult(
        atoms=working,
        converged=bool(converged),
        steps=optimizer.get_number_of_steps(),
        fmax=float(np.sqrt((forces**2).sum(axis=1)).max()),
        initial_energy=initial_energy,
        final_energy=working.get_potential_energy(),
        spacegroup=(initial_group, _spacegroup(working, symprec)),
        trajectory=trajectory,
    )
    write_amaterasu_structure(
        working, directory / f"{config.output.prefix}_relaxed.str", comment="relaxed by amaterasu"
    )
    return result


def _spacegroup(atoms: Atoms, symprec: float) -> int:
    try:
        return analyze(atoms, symprec=symprec).number
    except RuntimeError:  # pragma: no cover - defensive
        return 0


def _pulay_warning(pressure: float) -> str:
    return (
        "  NOTE: relaxing the cell. The stress includes a Pulay term from the finite "
        "plane-wave cutoff, so the equilibrium volume is biased until the cutoff is "
        "converged -- typically towards smaller volumes. Check with "
        "amaterasu.properties.stress.pulay_stress_estimate, or re-relax the ions at the "
        f"final fixed cell. Target pressure {pressure:g} GPa."
    )
