"""Everything a plane-wave calculation fixes once, before the SCF loop starts.

A :class:`PlaneWaveSystem` holds the geometry-dependent quantities that never change
during an SCF cycle: the basis (FFT grid, density sphere, one wavefunction sphere per
k-point), the Kleinman-Bylander projectors, the local ionic potential, the frozen partial
core density, the Ewald energy, and the symmetry operations used to symmetrise the
density. Only the density, the potential and the wavefunctions vary inside the loop.

Splitting it this way keeps the expensive setup -- the radial Fourier transforms, which
depend on ``|G|`` alone -- out of the iteration, and gives the later phases a single
object to rebuild when the geometry moves.

Units are Hartree atomic units throughout: the cell and positions are converted from the
Angstrom of :class:`ase.Atoms` exactly once, here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

from ..basis.fft import FFTGrid, symmetry_compatible_multiple
from ..basis.gvectors import GSphere
from ..config.schema import Config
from ..pseudo.base import NCPseudopotential
from ..pseudo.kb import KBProjectors, build_projectors
from ..pseudo.paw.dataset import PAWDataset
from ..system import kpoints as kpoint_module
from ..system.symmetry import SymmetryInfo, analyze, spin_decorated, symmetrize_sphere
from ..units import ANG_IN_BOHR
from ..xc.driver import XCFunctional
from .ewald import ewald_energy
from .local import atomic_core_density, atomic_valence_density, local_potential
from .paw import (
    PAWMachinery,
    build_paw_machinery,
    build_paw_projectors,
    make_setups,
    vbar_potential,
)

__all__ = ["PlaneWaveSystem", "build_system", "default_band_count"]


def default_band_count(nelectrons: float, nspin: int, moment: float = 0.0) -> int:
    """Number of bands to solve for when the input does not say.

    A metal needs empty states above the Fermi level for the smearing to act on, and an
    iterative eigensolver converges the topmost band worst, so a margin is standard
    practice: 20% extra, and never fewer than four extra bands. For a spin-polarised run
    the majority channel holds ``(N + m)/2`` states, so the moment enters too.
    """
    occupied = 0.5 * nelectrons
    if nspin == 2:
        occupied += 0.5 * abs(moment)
    return max(math.ceil(occupied) + 4, math.ceil(1.2 * occupied))


@dataclass
class PlaneWaveSystem:
    """The fixed part of a plane-wave calculation."""

    atoms: Atoms
    #: Direct cell in Bohr (rows).
    cell: np.ndarray
    #: Cartesian positions in Bohr.
    positions: np.ndarray
    symbols: list[str]
    pseudopotentials: dict[str, NCPseudopotential | PAWDataset]
    xc: XCFunctional

    ecut: float
    ecut_rho: float
    grid: FFTGrid
    density_sphere: GSphere
    kpoints: kpoint_module.KPointSet
    wave_spheres: list[GSphere]
    projectors: list[KBProjectors]

    nspin: int
    nbands: int
    nelectrons: float
    initial_magmoms: np.ndarray

    ewald: float
    v_local: np.ndarray
    core_density: np.ndarray | None
    symmetry: SymmetryInfo | None = None
    #: Present only for a PAW calculation; ``None`` for norm-conserving.
    paw: PAWMachinery | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_paw(self) -> bool:
        return self.paw is not None

    @property
    def volume(self) -> float:
        """Cell volume in Bohr^3."""
        return self.grid.volume

    @property
    def nkpoints(self) -> int:
        return len(self.kpoints)

    def __str__(self) -> str:
        npw = [len(s) for s in self.wave_spheres]
        lines = [
            f"basis                {self.grid}",
            f"  wavefunctions      ecut={self.ecut:g} Ha ({2 * self.ecut:g} Ry), "
            f"{min(npw)}-{max(npw)} plane waves per k-point",
            f"  density            ecut={self.ecut_rho:g} Ha ({2 * self.ecut_rho:g} Ry), "
            f"{len(self.density_sphere)} plane waves",
            f"  k-points           {self.kpoints}",
            f"  bands              {self.nbands} for {self.nelectrons:g} electrons"
            f"{' (spin-polarised)' if self.nspin == 2 else ''}",
            f"  projectors         {len(self.projectors[0])} "
            f"({'symmetrised density' if self.symmetry else 'no symmetrisation'})",
        ]
        return "\n".join(lines)

    def smooth_electron_count(self, occupancies: np.ndarray | None = None) -> float:
        """How many electrons the *smooth* valence density holds.

        Not the same as the number of valence electrons. The PAW overlap operator makes
        ``integral |psi~|^2 = 1 - sum_ij D_ij O_ij`` rather than 1, so the smooth density
        is short by exactly the augmentation charge, which the compensation charge and the
        one-centre terms put back. For a norm-conserving run the two are the same number.

        Getting this wrong is not a slow start but a stalled one: Kerker's factor is
        exactly zero at ``G = 0``, so the mixer conserves the charge of whatever it is
        given. A starting density normalised to the full valence charge therefore keeps a
        constant excess forever, and the density residual settles at exactly the
        augmentation charge per electron and never moves.
        """
        if self.paw is None:
            return self.nelectrons
        if occupancies is None:
            from ..scf.driver import _initial_occupancies

            occupancies = _initial_occupancies(self)
        return self.nelectrons - float(np.einsum("sij,ij->", occupancies, self.paw.overlap))

    def initial_density(self) -> np.ndarray:
        """Superposed atomic valence densities, split by the initial moments.

        Returns shape ``(nspin, *grid.shape)`` in electrons/Bohr^3.
        """
        total = atomic_valence_density(
            self.grid,
            self.density_sphere,
            self.positions,
            self.symbols,
            self.pseudopotentials,
            total_electrons=self.smooth_electron_count(),
        )
        if self.nspin == 1:
            return total[None, ...]

        # Build a moment density m(r) that integrates to the requested total moment, and
        # split as (n +- m)/2. Each atom contributes its own density scaled by
        # magmom / Z_valence, so an atom asked for 2.2 muB out of 16 valence electrons
        # starts 14% polarised rather than fully polarised.
        #
        # Getting this wrong is not a slow start but a different calculation: a fully
        # polarised iron atom is 16 muB, and an SCF launched from there swings by tens of
        # eV per iteration and may not come back.
        moment_density = np.zeros_like(total)
        for atom, symbol in enumerate(self.symbols):
            if abs(self.initial_magmoms[atom]) < 1e-12:
                continue
            single = atomic_valence_density(
                self.grid,
                self.density_sphere,
                self.positions[atom : atom + 1],
                [symbol],
                self.pseudopotentials,
                total_electrons=None,
            )
            charge = float(np.sum(single)) * self.grid.cell_volume_per_point
            if charge <= 0:  # pragma: no cover - a dataset with no valence density
                continue
            moment_density += (self.initial_magmoms[atom] / charge) * single

        # No point may end up with a negative spin density, which caps |m| at n.
        moment_density = np.clip(moment_density, -total, total)
        return np.stack([0.5 * (total + moment_density), 0.5 * (total - moment_density)])

    def symmetrize_density(self, density: np.ndarray) -> np.ndarray:
        """Average a density over the space group, one spin channel at a time.

        Required whenever the Brillouin-zone sum runs over the irreducible wedge: the
        wedge sum alone has only the symmetry of the wedge.
        """
        if self.symmetry is None or self.symmetry.n_operations <= 1:
            return density
        out = np.empty_like(density)
        for spin in range(density.shape[0]):
            transformed = self.grid.to_reciprocal(density[spin].astype(complex))
            coefficients = self.grid.gather(transformed, self.density_sphere)
            averaged = symmetrize_sphere(
                coefficients,
                self.density_sphere,
                self.grid,
                self.symmetry.rotations,
                self.symmetry.translations,
            )
            out[spin] = self.grid.to_real(self.grid.scatter(averaged, self.density_sphere)).real
        return out


def build_system(
    atoms: Atoms,
    config: Config,
    pseudopotentials: dict[str, NCPseudopotential | PAWDataset],
    ecut: float,
    ecut_rho: float,
) -> PlaneWaveSystem:
    """Assemble the fixed part of a calculation from a validated input.

    ``ecut`` and ``ecut_rho`` are in Hartree, already resolved from the input or the
    dataset by :func:`amaterasu.pseudo.registry.resolve_cutoffs`.
    """
    if not all(atoms.pbc):
        raise ValueError("amaterasu solves 3D-periodic systems; every direction must be periodic")
    cell = np.asarray(atoms.cell[:], dtype=float) * ANG_IN_BOHR
    positions = np.asarray(atoms.get_positions(), dtype=float) * ANG_IN_BOHR
    symbols = list(atoms.get_chemical_symbols())
    missing = sorted(set(symbols) - set(pseudopotentials))
    if missing:
        raise ValueError(f"no pseudopotential supplied for {', '.join(missing)}")

    nspin = 2 if config.system.spin == "polarized" else 1
    magmoms = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float).reshape(len(atoms), -1)
    magmoms = magmoms[:, 0]
    nelectrons = sum(pseudopotentials[s].valence_charge for s in symbols) - config.system.charge
    if nelectrons <= 0:
        raise ValueError(
            f"the cell holds {nelectrons:g} electrons; check system.charge against the "
            "valence charges of the datasets"
        )

    # Symmetry (and therefore the irreducible wedge) must see the magnetic decoration, or
    # an antiferromagnet is reduced as if it were non-magnetic.
    reference = spin_decorated(atoms, magmoms) if nspin == 2 else atoms
    warnings: list[str] = []
    try:
        group = analyze(reference, symprec=config.kpoints.symprec)
    except RuntimeError as exc:  # pragma: no cover - spglib failure is rare
        warnings.append(f"symmetry analysis failed ({exc}); running without symmetrisation")
        group = None
    # kpoints.symmetry governs the Brillouin-zone reduction and the density averaging that
    # goes with it. The grid shape below deliberately does *not* depend on it: two systems
    # built for the same crystal must share a grid, so that a band-structure run at
    # explicit k-points can reuse the potential from a symmetry-reduced SCF.
    symmetry = group if config.kpoints.symmetry else None
    kpoints = kpoint_module.generate(reference, config.kpoints)

    # The grid is sized for the density cutoff and then made commensurate with the space
    # group's fractional translations, so that pointwise quantities -- v_xc above all --
    # are exactly invariant rather than invariant to within an aliasing error.
    multiple = symmetry_compatible_multiple(group.translations) if group else 1
    grid = FFTGrid(
        shape=FFTGrid.for_cutoff(cell, ecut_rho, multiple=multiple).shape,
        cell=cell,
        backend=config.parallel.fft_backend,
    )
    density_sphere = GSphere.build(cell, ecut_rho)

    wave_spheres = [GSphere.build(cell, ecut, kpoint=k) for k in kpoints.points]

    nbands = config.occupations.nbands or default_band_count(
        nelectrons, nspin, float(np.sum(np.abs(magmoms)))
    )
    smallest = min(len(s) for s in wave_spheres)
    if nbands > smallest:
        warnings.append(
            f"asked for {nbands} bands but the smallest k-point has only {smallest} plane "
            f"waves; reducing to {smallest}. Raise basis.ecut instead."
        )
        nbands = smallest

    # ------------------------------------------------------------------ PAW or not
    # The two methods differ in what the ionic terms *are*, not merely in their values.
    # A norm-conserving run has a local ionic pseudopotential with a Coulomb tail and a
    # separate Ewald sum for the ions; a PAW run has neither. The nuclear charge sits
    # inside the compensation charge, which makes the smooth density neutral, so the
    # plane-wave Hartree term is the whole of the electrostatics -- and what is left of
    # the local potential is the short-ranged zero potential v_bar.
    kinds = {getattr(p, "kind", "NC") for p in pseudopotentials.values()}
    if kinds == {"PAW"}:
        # Both PAW formats become a PAWDataset: PAW-XML directly, and a Kresse-Joubert
        # UPF through amaterasu.pseudo.readers.upfpaw. Anything that declares itself PAW
        # and did not is a file whose augmentation data never got parsed -- an ultrasoft
        # UPF read for its norm-conserving content only, or read_upf called directly on a
        # PAW file. Say plainly which one turned up instead of failing later on a missing
        # attribute.
        partial = {
            symbol: getattr(data, "source", "?")
            for symbol, data in pseudopotentials.items()
            if not isinstance(data, PAWDataset)
        }
        if partial:
            listing = ", ".join(f"{symbol} ({source})" for symbol, source in partial.items())
            raise NotImplementedError(
                f"these datasets declare PAW but carry no augmentation data that was "
                f"read, so they cannot drive a PAW run: {listing}. Load them through "
                f"amaterasu.pseudo.readers.read_pseudopotential, which routes a PAW UPF "
                f"to the Kresse-Joubert reader, or set pseudo.method to nc."
            )
        setups = make_setups(
            pseudopotentials,
            XCFunctional(config.xc.functional),
            shape=config.pseudo.compensation_shape,
            shape_scale=config.pseudo.compensation_scale,
        )
        paw = build_paw_machinery(
            grid,
            density_sphere,
            positions,
            symbols,
            setups,
            atoms=reference,
            symmetry=symmetry,
            cell=cell,
        )
        warnings.extend(paw.warnings)
        projectors = [
            build_paw_projectors(sphere, positions, symbols, setups, grid.volume)
            for sphere in wave_spheres
        ]
        v_local = vbar_potential(grid, density_sphere, positions, symbols, setups)
        ewald = 0.0
    elif kinds == {"NC"}:
        paw = None
        projectors = [
            build_projectors(sphere, positions, symbols, pseudopotentials, grid.volume)
            for sphere in wave_spheres
        ]
        v_local = local_potential(grid, density_sphere, positions, symbols, pseudopotentials)
        charges = np.array([pseudopotentials[s].valence_charge for s in symbols])
        ewald = ewald_energy(cell, positions, charges)
    else:
        raise NotImplementedError(
            f"the cell mixes {' and '.join(sorted(kinds))} datasets; amaterasu needs one "
            "method for the whole cell because the ionic energy is partitioned differently"
        )

    system = PlaneWaveSystem(
        atoms=atoms,
        cell=cell,
        positions=positions,
        symbols=symbols,
        pseudopotentials=pseudopotentials,
        xc=XCFunctional(config.xc.functional),
        ecut=ecut,
        ecut_rho=ecut_rho,
        grid=grid,
        density_sphere=density_sphere,
        kpoints=kpoints,
        wave_spheres=wave_spheres,
        projectors=projectors,
        nspin=nspin,
        nbands=nbands,
        nelectrons=float(nelectrons),
        initial_magmoms=magmoms,
        ewald=ewald,
        v_local=v_local,
        core_density=atomic_core_density(
            grid, density_sphere, positions, symbols, pseudopotentials
        ),
        symmetry=symmetry,
        paw=paw,
        warnings=warnings,
    )
    return system
