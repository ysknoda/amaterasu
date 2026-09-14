"""Molecular dynamics, driven through ``ase.md``.

As with the relaxation driver, the integrators, thermostats and barostats are ASE's; this
module maps the YAML settings onto them and monitors the trajectory. What amaterasu
contributes is the force at each step.

Which energy the forces belong to
---------------------------------
With smeared occupations the force is the gradient of the **free energy** ``F = E - sigma S``,
not of ``E`` and not of the ``sigma -> 0`` extrapolation ``E0``. A dynamics run therefore
conserves ``F + E_kin``, and it is that combination the drift monitor watches. Using ``E0``
instead looks almost right -- the difference is half the entropy term -- and produces a
slow, systematic energy drift that is easily mistaken for a timestep problem.

Ensembles
---------
``nve``  VelocityVerlet -- the reference for testing, since nothing is supposed to change.
``nvt``  Nose-Hoover chain (deterministic, correct canonical sampling), Bussi (stochastic
         velocity rescaling), Langevin, or Andersen.
``npt``  Isotropic MTK (Nose-Hoover chains on both the thermostat and the barostat),
         Parrinello-Rahman, or Berendsen -- the last of which does *not* sample a correct
         ensemble and is offered only for fast equilibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase import units as ase_units
from ase.md import Andersen, Bussi, IsotropicMTKNPT, Langevin, NoseHooverChainNVT, VelocityVerlet
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from ..calculator import Amaterasu
from ..config.schema import Config
from ..units import GPA_IN_EV_PER_ANG3

__all__ = ["MDResult", "run"]


@dataclass
class MDResult:
    """A finished trajectory and the statistics taken over its production part."""

    atoms: Atoms
    steps: int
    #: Per step, after equilibration: (time in s, potential, kinetic, total energy in eV,
    #: temperature in K, pressure in GPa).
    history: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)
    trajectory: Path | None = None

    @property
    def temperatures(self) -> np.ndarray:
        return np.array([row[4] for row in self.history])

    @property
    def total_energies(self) -> np.ndarray:
        return np.array([row[3] for row in self.history])

    def energy_spread(self, natoms: int) -> float:
        """Peak-to-peak variation of the conserved energy, in meV/atom.

        This is the diagnostic that means something at *any* trajectory length. The
        conserved quantity of a Verlet integrator oscillates at the vibrational
        frequencies with an amplitude set by the timestep, and drifts on top of that only
        if the forces are inconsistent with the energy. A spread far below the kinetic
        energy scale says the integration is sound; a growing one says it is not.
        """
        if not self.history:  # pragma: no cover - a zero-step run
            return float("nan")
        energies = self.total_energies
        return 1000.0 * float(energies.max() - energies.min()) / natoms

    def energy_drift(self, natoms: int) -> float:
        """Least-squares *secular* drift of the conserved energy, in meV/atom/ps.

        Only meaningful over several vibrational periods. Fitting a straight line to less
        than one period measures the slope of the oscillation instead: on a 2-atom silicon
        cell, whose optical mode has a 64 fs period, the same trajectory gives
        +1.4 meV/atom/ps at 30 fs and -0.5 meV/atom/ps at 40 fs while the total energy
        never moves by more than 0.08 meV/atom. Use :meth:`energy_spread` for short runs,
        and this only once the trajectory covers a few periods.
        """
        if len(self.history) < 3:
            return float("nan")
        times = np.array([row[0] for row in self.history]) * 1e12  # picoseconds
        slope = np.polyfit(times, self.total_energies, 1)[0]
        return 1000.0 * slope / natoms

    def render(self, natoms: int) -> str:
        if not self.history:  # pragma: no cover - a zero-step run
            return "\nno production steps were recorded"
        temperature = self.temperatures
        pressure = np.array([row[5] for row in self.history])
        lines = [
            "",
            f"molecular dynamics finished after {self.steps} steps",
            f"  {'production steps':<26s} {len(self.history):16d}",
            f"  {'mean temperature':<26s} {temperature.mean():16.4f} +- {temperature.std():.4f} K",
            f"  {'mean total energy':<26s} {self.total_energies.mean():16.9f} eV",
            f"  {'conserved energy spread':<26s} {self.energy_spread(natoms):16.6f} meV/atom",
            f"  {'energy drift':<26s} {self.energy_drift(natoms):16.6f} meV/atom/ps"
            f"   (needs several vibrational periods to mean anything)",
        ]
        if np.any(np.isfinite(pressure)):
            lines.append(
                f"  {'mean pressure':<26s} {np.nanmean(pressure):16.6f} +- "
                f"{np.nanstd(pressure):.6f} GPa"
            )
        return "\n".join(lines)


def _make_dynamics(atoms: Atoms, settings, timestep: float):
    """Build the ASE integrator named in ``md.ensemble``/``thermostat``/``barostat``."""
    temperature = settings.temperature
    if settings.ensemble == "nve":
        return VelocityVerlet(atoms, timestep)

    tdamp = settings.tdamp * ase_units.fs
    if settings.ensemble == "nvt":
        if settings.thermostat == "nose-hoover-chain":
            return NoseHooverChainNVT(atoms, timestep, temperature_K=temperature, tdamp=tdamp)
        if settings.thermostat == "bussi":
            return Bussi(atoms, timestep, temperature_K=temperature, taut=tdamp)
        if settings.thermostat == "langevin":
            friction = settings.friction / ase_units.fs
            return Langevin(atoms, timestep, temperature_K=temperature, friction=friction)
        return Andersen(atoms, timestep, temperature_K=temperature, andersen_prob=settings.friction)

    pressure_au = settings.pressure * GPA_IN_EV_PER_ANG3
    pdamp = settings.pdamp * ase_units.fs
    if settings.barostat == "mtk":
        return IsotropicMTKNPT(
            atoms,
            timestep,
            temperature_K=temperature,
            pressure_au=pressure_au,
            tdamp=tdamp,
            pdamp=pdamp,
        )
    if settings.barostat == "parrinello-rahman":
        return NPT(
            atoms,
            timestep,
            temperature_K=temperature,
            externalstress=pressure_au,
            ttime=tdamp,
            pfactor=pdamp**2 * ase_units.GPa * 100.0,
        )
    return NPTBerendsen(
        atoms, timestep, temperature_K=temperature, pressure_au=pressure_au, taut=tdamp, taup=pdamp
    )


def run(
    atoms: Atoms,
    config: Config,
    *,
    output_directory: Path | None = None,
    logger=print,
    download: bool = True,
) -> MDResult:
    """Run molecular dynamics according to ``config.md``."""
    settings = config.md
    working = atoms.copy()
    working.calc = Amaterasu(config, logger=logger, download=download)

    rng = np.random.default_rng(settings.seed)
    MaxwellBoltzmannDistribution(
        working, temperature_K=settings.temperature, rng=np.random.RandomState(rng.integers(2**31))
    )
    # A drifting centre of mass is a rigid translation: it carries kinetic energy that
    # never thermalises and biases every temperature estimate.
    Stationary(working)

    timestep = settings.timestep * ase_units.fs
    dynamics = _make_dynamics(working, settings, timestep)

    directory = Path(output_directory or config.output.directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    trajectory = directory / f"{config.output.prefix}_md.traj"
    from ase.io.trajectory import Trajectory

    writer = Trajectory(str(trajectory), "w", working)
    dynamics.attach(writer.write, interval=1)

    result = MDResult(atoms=working, steps=0, trajectory=trajectory)
    wants_pressure = settings.ensemble == "npt"
    logger(
        f"  {'step':>6s} {'time (s)':>12s} {'E_pot (eV)':>18s} {'E_kin (eV)':>12s} "
        f"{'E_tot (eV)':>18s} {'T (K)':>10s}"
    )

    def record():
        step = dynamics.get_number_of_steps()
        # The force-consistent energy is the one the dynamics conserves; see the note above.
        potential = working.get_potential_energy(force_consistent=True)
        kinetic = working.get_kinetic_energy()
        temperature = working.get_temperature()
        pressure = float("nan")
        if wants_pressure:
            pressure = working.calc.get_pressure(working)
        logger(
            f"  {step:6d} {step * timestep / ase_units.fs * 1e-15:12.4e} {potential:18.9f} "
            f"{kinetic:12.6f} {potential + kinetic:18.9f} {temperature:10.3f}"
        )
        if step >= settings.equilibration_steps:
            result.history.append(
                (
                    step * settings.timestep * 1e-15,
                    potential,
                    kinetic,
                    potential + kinetic,
                    temperature,
                    pressure,
                )
            )

    dynamics.attach(record, interval=1)
    dynamics.run(settings.steps)
    writer.close()
    result.steps = dynamics.get_number_of_steps()
    return result
