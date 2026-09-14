"""The ASE calculator interface: the single point where amaterasu meets the outside world.

Everything ASE can drive -- geometry optimisation, molecular dynamics, nudged elastic
band, the phonon finite-difference machinery -- talks to a
:class:`ase.calculators.calculator.Calculator`. Implementing that one class correctly is
therefore what turns the self-consistent field solver into a usable code, and it is why
amaterasu does not reimplement any of those algorithms.

Reusing the previous solution
-----------------------------
A relaxation or an MD trajectory asks for hundreds of calculations at geometries that
differ by a fraction of a Bohr. Starting each from the superposed atomic densities throws
away almost all of the previous answer, so the last density and orbitals are carried over
as the next starting guess whenever they still *fit*: the same grid, the same number of
k-points with the same plane-wave counts, and the same spin and band counts.

Fitting is the whole test, and it has to be checked against the assembled system rather
than guessed from the input. Displacing one atom of a symmetric crystal lowers its space
group, which enlarges the irreducible wedge and changes both the number of k-points and
the size of each sphere -- so "the cell and the cutoff are unchanged" is *not* sufficient.
When the shapes do fit, reuse is safe even across a small cell change, because a starting
guess only has to be close; the density is rescaled by the volume ratio so that it still
holds the right number of electrons.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .config.schema import Config
from .hamiltonian.system import PlaneWaveSystem, build_system
from .properties.forces import compute_forces
from .pseudo.registry import load_pseudopotentials, resolve_cutoffs
from .scf.driver import SCFResult, run_scf
from .units import (
    BOHR_IN_ANG,
    HARTREE_IN_EV,
    HARTREE_PER_BOHR3_IN_GPA,
    HARTREE_PER_BOHR_IN_EV_PER_ANG,
)

__all__ = ["Amaterasu"]

#: eV/Angstrom^3 per Ha/Bohr^3 -- the unit ASE reports stress in.
_STRESS_TO_ASE = HARTREE_IN_EV / BOHR_IN_ANG**3


class Amaterasu(Calculator):
    """A plane-wave DFT calculator for :class:`ase.Atoms`.

    Parameters
    ----------
    config
        A validated :class:`~amaterasu.config.schema.Config`. Anything ASE needs that the
        config does not carry (the structure path, the solver) is ignored here.
    logger
        Called with each line of SCF output; ``None`` is silent.
    download
        Fetch missing pseudopotential datasets. Turn it off for offline runs.
    """

    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "magmom",
        "magmoms",
    ]

    def __init__(
        self,
        config: Config | None = None,
        *,
        logger=None,
        download: bool = True,
        directory: str | Path = ".",
        **kwargs,
    ):
        super().__init__(directory=str(directory), **kwargs)
        self.config = config or Config(structure="-", solver="scf")
        self.logger = logger or (lambda *_: None)
        self.download = download
        self.system: PlaneWaveSystem | None = None
        self.scf: SCFResult | None = None
        self._pseudopotentials = None
        self._cutoffs: tuple[float, float] | None = None
        self._previous_volume: float | None = None
        #: Number of SCF iterations spent on each call; a cheap way to see whether the
        #: restart is doing its job over a trajectory.
        self.iteration_history: list[int] = []

    # ------------------------------------------------------------------ internals
    def _load_datasets(self, atoms) -> None:
        if self._pseudopotentials is not None:
            return
        symbols = sorted(set(atoms.get_chemical_symbols()))
        resolved = load_pseudopotentials(self.config, symbols, download=self.download)
        for warning in resolved.warnings:
            self.logger(f"  WARNING: {warning}")
        self._pseudopotentials = resolved.pseudopotentials
        self._cutoffs = resolve_cutoffs(self.config, resolved)

    def _starting_guess(self, system: PlaneWaveSystem):
        """The previous density and orbitals if they fit ``system``, else ``(None, None)``."""
        if self.scf is None:
            return None, None
        density = self.scf.density
        if density.shape != (system.nspin, *system.grid.shape):
            return None, None
        # The density is a field per unit volume; a changed cell must not change the
        # electron count it carries.
        if self._previous_volume:
            density = density * (self._previous_volume / system.grid.volume)

        wavefunctions = self.scf.wavefunctions
        fits = len(wavefunctions) == system.nspin and all(
            len(per_spin) == system.nkpoints
            and all(
                block.shape == (system.nbands, len(sphere))
                for block, sphere in zip(per_spin, system.wave_spheres, strict=True)
            )
            for per_spin in wavefunctions
        )
        return density, (wavefunctions if fits else None)

    # -------------------------------------------------------------------- ASE hook
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        self._load_datasets(atoms)
        ecut, ecut_rho = self._cutoffs

        self.system = build_system(atoms, self.config, self._pseudopotentials, ecut, ecut_rho)
        for warning in self.system.warnings:
            self.logger(f"  WARNING: {warning}")
        self._warn_about_core_ringing()
        density, wavefunctions = self._starting_guess(self.system)

        result = run_scf(
            self.system,
            self.config,
            logger=self.logger,
            density=density,
            wavefunctions=wavefunctions,
        )
        if not result.converged:
            self.logger(
                f"  WARNING: the SCF stopped after {result.iterations} iterations without "
                "meeting both convergence criteria; the forces below are not trustworthy"
            )
        self.scf = result
        self._previous_volume = self.system.grid.volume
        self.iteration_history.append(result.iterations)

        self.results = {
            "energy": result.energies.extrapolated * HARTREE_IN_EV,
            "free_energy": result.energies.free_energy * HARTREE_IN_EV,
            "magmom": result.magnetization,
            "magmoms": np.zeros(len(atoms)),
        }
        if "forces" in properties or "stress" in properties:
            forces = compute_forces(
                self.system,
                result.density,
                result.wavefunctions,
                result.occupations,
                eigenvalues=result.eigenvalues,
            )
            self.results["forces"] = forces * HARTREE_PER_BOHR_IN_EV_PER_ANG
        if "stress" in properties:
            from .properties.stress import compute_stress

            stress = compute_stress(
                self.system,
                result.density,
                result.wavefunctions,
                result.occupations,
                eigenvalues=result.eigenvalues,
            )
            self.results["stress"] = _to_voigt(stress * _STRESS_TO_ASE)

    def _warn_about_core_ringing(self) -> None:
        from .hamiltonian.local import core_density_ringing

        ringing = core_density_ringing(self.system.core_density)
        if ringing > 1e-3:
            self.logger(
                f"  WARNING: the partial core density dips {ringing:.1%} below zero, which "
                "means basis.ecut_rho does not resolve it. Raise the cutoff: forces and "
                "stresses through the core correction are unreliable at this resolution"
            )

    # ------------------------------------------------------------------ extras
    def get_pressure(self, atoms=None) -> float:
        """Hydrostatic pressure in GPa: ``-tr(sigma)/3``, positive when the cell would expand."""
        stress = self.get_stress(atoms)
        return -float(np.mean(stress[:3])) / _STRESS_TO_ASE * HARTREE_PER_BOHR3_IN_GPA

    def get_band_structure_data(self) -> dict:
        """Eigenvalues, occupations and the Fermi level in eV, for post-processing."""
        if self.scf is None:  # pragma: no cover - defensive
            raise RuntimeError("no calculation has been run yet")
        return {
            "eigenvalues": self.scf.eigenvalues * HARTREE_IN_EV,
            "occupations": self.scf.occupations,
            "fermi_level": self.scf.fermi_level * HARTREE_IN_EV,
            "kpoints": self.system.kpoints.points,
            "weights": self.system.kpoints.weights,
            "gap": None if self.scf.gap is None else self.scf.gap * HARTREE_IN_EV,
        }


def _to_voigt(tensor: np.ndarray) -> np.ndarray:
    """ASE's Voigt order: ``[xx, yy, zz, yz, xz, xy]``."""
    return np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            0.5 * (tensor[1, 2] + tensor[2, 1]),
            0.5 * (tensor[0, 2] + tensor[2, 0]),
            0.5 * (tensor[0, 1] + tensor[1, 0]),
        ]
    )
