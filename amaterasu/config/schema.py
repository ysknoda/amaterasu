"""The amaterasu calculation input schema.

The whole YAML input is described here as a tree of dataclasses with ordinary type hints;
:mod:`amaterasu.config._loader` turns a parsed YAML mapping into this tree and reports
every problem at once. Sections that are irrelevant to the chosen ``solver`` are still
parsed and validated, so a single ``calc.yaml`` can be reused across solvers by changing
only the ``solver:`` line.

Units
-----
Fields typed ``float | str`` accept an explicit unit (``"816 eV"``, ``"60 Ry"``) and fall
back to the documented default unit for a bare number. See :mod:`amaterasu.units`.
Fields typed plain ``float`` use the unit named in their comment and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "BandsConfig",
    "BasisConfig",
    "Config",
    "DFPTConfig",
    "DOSConfig",
    "HubbardConfig",
    "HubbardSpecies",
    "KpointsConfig",
    "MDConfig",
    "MixingConfig",
    "NEBConfig",
    "OccupationsConfig",
    "OutputConfig",
    "ParallelConfig",
    "PhononConfig",
    "PseudoConfig",
    "RelaxConfig",
    "SCFConfig",
    "SystemConfig",
    "WannierConfig",
    "XCConfig",
]

Energy = float | str  # a number in the field's default unit, or "<value> <unit>"


# --------------------------------------------------------------------------- system
@dataclass
class HubbardSpecies:
    """Hubbard parameters for one species. ``U``/``J`` in eV."""

    U: float = 0.0
    J: float = 0.0
    #: Shell the correction acts on. ``None`` picks the element's open valence shell
    #: (d for transition metals, f for lanthanides/actinides).
    orbital: Literal["s", "p", "d", "f"] | None = None


@dataclass
class HubbardConfig:
    """DFT+U settings.

    ``dudarev`` uses the single effective parameter U_eff = U - J (Dudarev et al.,
    PRB 57, 1505 (1998)); ``liechtenstein`` uses the rotationally invariant (U, J) form
    (Liechtenstein et al., PRB 52, R5467 (1995)). A U value is only meaningful together
    with the projector it was fitted for, so ``projector`` is recorded in the output.
    """

    scheme: Literal["none", "dudarev", "liechtenstein"] = "none"
    projector: Literal["paw", "atomic", "wannier"] = "atomic"
    species: dict[str, HubbardSpecies] = field(default_factory=dict)


@dataclass
class SystemConfig:
    spin: Literal["none", "polarized"] = "none"
    #: Net charge of the cell in units of |e| (positive = electrons removed). A charged
    #: cell is neutralised by a uniform background; finite-size corrections are not applied.
    charge: float = 0.0
    #: Validated, but DFT+U is not implemented yet (Phase 5): any ``scheme`` other than
    #: ``none`` is refused at load time.
    hubbard: HubbardConfig = field(default_factory=HubbardConfig)


@dataclass
class XCConfig:
    functional: Literal["LDA-PZ", "LDA-PW", "PBE", "PBEsol"] = "PBE"


# --------------------------------------------------------------------------- basis
@dataclass
class PseudoConfig:
    """Pseudopotential / PAW dataset selection.

    PBEsol PAW datasets exist only in UPF form (pslibrary ``*-kjpaw_psl``); the PAW-XML
    tables (JTH, GPAW) cover PBE and LDA only. Those UPF files are read -- they are written
    in the Kresse-Joubert formulation and converted on load, see
    :mod:`amaterasu.pseudo.readers.upfpaw` -- but they are not fetchable, so name one in
    ``files``. ``allow_xc_mismatch`` stays False by default because using a PBE dataset in a
    PBEsol run is a genuine inconsistency.
    """

    method: Literal["nc", "paw"] = "nc"
    family: str | None = None
    directory: str = "~/.amaterasu/pseudos"
    #: Explicit per-species filenames, overriding ``family``.
    files: dict[str, str] = field(default_factory=dict)
    allow_xc_mismatch: bool = False
    #: Shape carrying the PAW compensation charge. ``gaussian`` is the default because it
    #: is the setting the Phase 4 gates were measured in; ``bessel`` is compactly supported,
    #: which is the more defensible formulation, and is better on some measures and worse on
    #: others. :mod:`amaterasu.pseudo.paw.shape` has the full comparison. Ignored for ``nc``.
    compensation_shape: Literal["gaussian", "bessel"] = "gaussian"
    #: Dimensionless deformation of that shape, 1 being the default. The total energy should
    #: not depend on it and still does, by an amount that measures how well the dataset's
    #: partial waves span the smooth density inside the sphere; this knob is what measures
    #: that. See :func:`amaterasu.pseudo.paw.shape.make_shape`.
    compensation_scale: float = 1.0


@dataclass
class BasisConfig:
    """Basis set. ``ecut``/``ecut_rho`` default to Ry (the pslibrary/SSSP/QE convention)."""

    type: Literal["planewave", "mixed"] = "planewave"
    #: Wavefunction cutoff. ``None`` means "take the dataset's recommended value".
    ecut: Energy | None = None
    #: Density/potential cutoff. ``None`` means 4*ecut for norm-conserving, or the
    #: dataset's recommendation for PAW (whose augmentation charges are much harder).
    ecut_rho: Energy | None = None
    #: Mixed basis only: extra localized orbitals per species, e.g. ``{Ni: [3d]}``.
    atomic_orbitals: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class KpointsConfig:
    scheme: Literal["monkhorst-pack", "gamma", "density", "explicit"] = "monkhorst-pack"
    grid: list[int] = field(default_factory=lambda: [1, 1, 1])
    shift: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: For ``scheme: density`` -- maximum spacing between k-points, in 1/Angstrom.
    density: float | None = None
    #: For ``scheme: explicit`` -- fractional coordinates in the reciprocal cell.
    points: list[list[float]] | None = None
    weights: list[float] | None = None
    #: Reduce to the irreducible wedge using the crystal symmetry (spglib).
    symmetry: bool = True
    symprec: float = 1e-5


@dataclass
class OccupationsConfig:
    """Brillouin-zone integration. ``smearing`` defaults to eV."""

    scheme: Literal[
        "fermi-dirac",
        "gaussian",
        "methfessel-paxton",
        "marzari-vanderbilt",
        "tetrahedron",
        "fixed",
    ] = "gaussian"
    #: Smearing width. The default is deliberately narrow -- narrower than Quantum
    #: ESPRESSO's 0.136 eV or VASP's 0.2 eV -- because a wide smearing hides an
    #: under-converged k-point mesh rather than fixing it. A narrow width demands a denser
    #: mesh; widen it only after checking that the mesh is converged at the width you use.
    #: Use ``fixed`` for an insulator, where no smearing is needed at all.
    smearing: Energy = 0.05
    #: Expansion order for ``methfessel-paxton``.
    order: int = 1
    #: Number of bands; ``None`` derives it from the electron count with a margin.
    nbands: int | None = None


# --------------------------------------------------------------------------- SCF
@dataclass
class MixingConfig:
    algorithm: Literal["linear", "anderson", "pulay", "broyden"] = "pulay"
    beta: float = 0.4
    history: int = 8
    preconditioner: Literal["none", "kerker"] = "kerker"
    #: Kerker screening wavevector in 1/Bohr. Physically this stands in for the
    #: Thomas-Fermi wavevector ``k_TF = sqrt(4 k_F / pi)``, which for the metallic range
    #: ``r_s = 2-3`` is 0.90-1.11 1/Bohr. The default sits above that range, so the damping
    #: of long-wavelength charge sloshing is deliberately on the strong side: a stable
    #: default matters more than the last few iterations of speed, and the converged
    #: answer does not depend on it at all. Lower it towards 1.0 for a fast metal, or set
    #: ``preconditioner: none`` for a wide-gap insulator, which does not slosh.
    kerker_q0: float = 1.5


@dataclass
class SCFConfig:
    """Electronic minimisation.

    ``mixing`` (density/potential mixing with a Pulay or Broyden history) is the default
    and the most robust choice for metals and large cells. ``direct-lbfgs``/``direct-cg``
    minimise the energy directly over the wavefunctions on the Stiefel manifold, which is
    the electronic-structure counterpart of an L-BFGS/CG geometry optimisation.
    """

    method: Literal["mixing", "direct-lbfgs", "direct-cg"] = "mixing"
    eigensolver: Literal["davidson", "lobpcg", "rmm-diis"] = "davidson"
    mixing: MixingConfig = field(default_factory=MixingConfig)
    max_steps: int = 100
    #: Total-energy change between iterations, in eV/atom.
    tol_energy: float = 1e-6
    #: Density change between iterations, as ``integral |n_out - n_in| dr / N_electrons``:
    #: dimensionless, and the same size for a light element and a heavy one. An absolute
    #: measure such as the RMS on the grid is not comparable across systems -- iron with
    #: semicore states peaks four orders of magnitude above silicon, so the same RMS
    #: threshold is trivial for one and unreachable for the other.
    tol_density: float = 1e-7
    #: History length for the direct-minimisation methods.
    lbfgs_history: int = 10


# --------------------------------------------------------------------------- solvers
@dataclass
class RelaxConfig:
    """Geometry optimisation, driven through ``ase.optimize`` and ``ase.filters``."""

    algorithm: Literal["fire", "fire2", "lbfgs", "lbfgs-ls", "bfgs", "ode12r"] = "fire"
    #: Convergence threshold on the maximum force, in eV/Angstrom.
    fmax: float = 0.01
    max_steps: int = 200
    #: Relax the cell as well as the ions.
    cell: bool = False
    cell_filter: Literal["frechet", "unitcell", "exp", "strain"] = "frechet"
    #: External hydrostatic pressure in GPa (variable-cell runs only).
    pressure: float = 0.0
    #: Project out symmetry-breaking force/stress components during relaxation.
    fix_symmetry: bool = True


@dataclass
class MDConfig:
    """Molecular dynamics, driven through ``ase.md``."""

    ensemble: Literal["nve", "nvt", "npt"] = "nvt"
    thermostat: Literal["nose-hoover-chain", "bussi", "langevin", "andersen"] = "nose-hoover-chain"
    barostat: Literal["mtk", "parrinello-rahman", "berendsen"] = "mtk"
    #: Timestep in fs. Must resolve the fastest vibration (~1 fs for systems with H).
    timestep: float = 1.0
    steps: int = 1000
    temperature: float = 300.0
    #: Target pressure in GPa (NPT only).
    pressure: float = 0.0
    #: Thermostat and barostat coupling times in fs.
    tdamp: float = 100.0
    pdamp: float = 1000.0
    #: Langevin friction in 1/fs.
    friction: float = 0.01
    #: Seed for the initial Maxwell-Boltzmann velocities and any stochastic thermostat.
    seed: int | None = None
    #: Discard this many steps before accumulating production averages.
    equilibration_steps: int = 0


@dataclass
class NEBConfig:
    """Nudged elastic band, driven through ``ase.mep``. ``climb`` selects CI-NEB."""

    initial: str | None = None
    final: str | None = None
    n_images: int = 7
    climb: bool = True
    #: Spring constant in eV/Angstrom^2.
    spring: float = 0.1
    interpolation: Literal["idpp", "linear"] = "idpp"
    algorithm: Literal["fire", "fire2", "lbfgs", "lbfgs-ls", "bfgs", "ode12r"] = "fire"
    fmax: float = 0.05
    max_steps: int = 200


@dataclass
class BandsConfig:
    #: ``auto`` uses the seekpath standard path (Hinuma et al., Comput. Mater. Sci. 128,
    #: 140 (2017)); otherwise a string like ``"G-X-W-K-G-L"``.
    path: str = "auto"
    npoints: int = 200


@dataclass
class DOSConfig:
    method: Literal["tetrahedron", "gaussian"] = "tetrahedron"
    grid: list[int] = field(default_factory=lambda: [16, 16, 16])
    #: Energy window relative to the Fermi level, in eV; ``None`` spans the eigenvalues.
    emin: float | None = None
    emax: float | None = None
    npoints: int = 2000
    #: Gaussian broadening in eV (``method: gaussian`` only).
    width: float = 0.05
    #: Also compute the site- and orbital-projected DOS.
    projected: bool = True


@dataclass
class PhononConfig:
    method: Literal["finite-displacement", "dfpt"] = "finite-displacement"
    supercell: list[int] = field(default_factory=lambda: [2, 2, 2])
    #: Displacement amplitude in Angstrom (finite-displacement only).
    displacement: float = 0.01
    mesh: list[int] = field(default_factory=lambda: [24, 24, 24])
    #: Non-analytical correction for LO-TO splitting; needs Born charges from DFPT.
    nac: bool = False
    path: str = "auto"


@dataclass
class DFPTConfig:
    perturbations: list[Literal["efield", "phonon", "strain"]] = field(
        default_factory=lambda: ["efield"]
    )
    qpoints: list[int] = field(default_factory=lambda: [1, 1, 1])
    #: Convergence threshold on the Sternheimer residual.
    tol: float = 1e-10
    max_steps: int = 100


@dataclass
class WannierConfig:
    #: ``w90-files`` writes .amn/.mmn/.eig for Wannier90; ``native`` runs the built-in
    #: maximally-localized Wannier function minimisation.
    backend: Literal["w90-files", "native"] = "w90-files"
    nwann: int | None = None
    projections: dict[str, list[str]] = field(default_factory=dict)
    disentangle: bool = False
    #: Inner (frozen) and outer disentanglement windows, in eV relative to the Fermi level.
    frozen_window: list[float] | None = None
    outer_window: list[float] | None = None


# --------------------------------------------------------------------------- runtime
@dataclass
class ParallelConfig:
    #: Number of MPI groups over k-points; ``"auto"`` uses as many as divide evenly. MPI is
    #: not implemented yet, so only ``"auto"`` and 1 are accepted and the run is serial.
    kpoint_groups: int | Literal["auto"] = "auto"
    #: Number of MPI groups over bands (Phase 9, not implemented yet: only 1 is accepted).
    band_groups: int = 1
    fft_backend: Literal["scipy", "pyfftw"] = "scipy"
    #: Use the numba-compiled kernels (Phase 9, not implemented yet: only False is
    #: accepted). False runs the numpy reference paths.
    numba: bool = False


@dataclass
class OutputConfig:
    directory: str = "./out"
    prefix: str = "amaterasu"
    save_wavefunctions: bool = False
    #: Verbosity of the command-line log. ``info`` prints the run header and the progress of
    #: every iteration; ``warning`` prints only warnings and the final result; ``error``
    #: prints nothing but errors and the final result. ``debug`` currently prints the same
    #: as ``info``.
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    #: Write a restart checkpoint every N ionic steps; 0 disables checkpointing. Not
    #: implemented yet, so only 0 is accepted.
    checkpoint_every: int = 0


# --------------------------------------------------------------------------- root
@dataclass
class Config:
    """A complete amaterasu calculation input."""

    structure: str
    solver: Literal["scf", "relax", "md", "neb", "bands", "dos", "phonon", "dfpt", "wannier"] = (
        "scf"
    )
    system: SystemConfig = field(default_factory=SystemConfig)
    xc: XCConfig = field(default_factory=XCConfig)
    pseudo: PseudoConfig = field(default_factory=PseudoConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    kpoints: KpointsConfig = field(default_factory=KpointsConfig)
    occupations: OccupationsConfig = field(default_factory=OccupationsConfig)
    scf: SCFConfig = field(default_factory=SCFConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    md: MDConfig = field(default_factory=MDConfig)
    neb: NEBConfig = field(default_factory=NEBConfig)
    bands: BandsConfig = field(default_factory=BandsConfig)
    dos: DOSConfig = field(default_factory=DOSConfig)
    phonon: PhononConfig = field(default_factory=PhononConfig)
    dfpt: DFPTConfig = field(default_factory=DFPTConfig)
    wannier: WannierConfig = field(default_factory=WannierConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
