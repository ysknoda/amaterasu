"""Cross-field validation and derived defaults for the configuration.

The type-driven pass in :mod:`amaterasu.config._loader` checks that every field has the
right type and an allowed value. This module checks the things a type cannot express:
relationships *between* fields, physically meaningless combinations, and settings that
are silently useless. As everywhere else in the input layer, problems are accumulated and
reported together.
"""

from __future__ import annotations

from ..units import parse_energy
from ._loader import ErrorLog
from .schema import Config

__all__ = ["resolve_and_validate"]

#: Solvers that need a second structure to interpolate between.
_TWO_ENDPOINT_SOLVERS = frozenset({"neb"})


def _check_positive_int_triplet(values, name: str, path: str, log: ErrorLog) -> None:
    if values is None:
        return
    if len(values) != 3:
        log.add(path, f"{name} needs exactly 3 integers, got {len(values)}: {values}")
        return
    bad = [v for v in values if not isinstance(v, int) or v < 1]
    if bad:
        log.add(path, f"{name} entries must be positive integers, got {values}")


def _check_energies(config: Config, log: ErrorLog) -> None:
    """Parse the unit-bearing energy fields and check their ordering."""
    ecut = ecut_rho = None
    for path, value, default_unit in (
        ("basis.ecut", config.basis.ecut, "Ry"),
        ("basis.ecut_rho", config.basis.ecut_rho, "Ry"),
        ("occupations.smearing", config.occupations.smearing, "eV"),
    ):
        if value is None:
            continue
        try:
            parsed = parse_energy(value, default_unit)
        except (TypeError, ValueError) as exc:
            log.add(path, str(exc))
            continue
        if parsed <= 0.0:
            log.add(path, f"must be positive, got {value!r}")
            continue
        if path == "basis.ecut":
            ecut = parsed
        elif path == "basis.ecut_rho":
            ecut_rho = parsed

    if ecut is not None and ecut_rho is not None and ecut_rho < ecut:
        log.add(
            "basis.ecut_rho",
            f"must be at least basis.ecut ({ecut_rho:.3f} Ha < {ecut:.3f} Ha); the density "
            "varies twice as fast as the wavefunctions, so its cutoff is never smaller",
        )

    if config.basis.ecut is None and config.pseudo.family is None and not config.pseudo.files:
        log.add(
            "basis.ecut",
            "is required unless pseudo.family or pseudo.files is set (from which a "
            "recommended cutoff can be read). The cutoff is the primary convergence "
            "parameter and is never guessed",
        )


def _check_kpoints(config: Config, log: ErrorLog) -> None:
    k = config.kpoints
    if k.scheme in ("monkhorst-pack", "density"):
        _check_positive_int_triplet(k.grid, "kpoints.grid", "kpoints.grid", log)
    if len(k.shift) != 3:
        log.add("kpoints.shift", f"needs exactly 3 numbers, got {len(k.shift)}")
    if k.scheme == "density" and k.density is None:
        log.add("kpoints.density", "is required when kpoints.scheme is 'density'")
    if k.scheme == "density" and k.density is not None and k.density <= 0:
        log.add("kpoints.density", f"must be positive, got {k.density}")
    if k.scheme == "explicit":
        if not k.points:
            log.add("kpoints.points", "is required when kpoints.scheme is 'explicit'")
        elif k.weights is not None and len(k.weights) != len(k.points):
            log.add(
                "kpoints.weights",
                f"has {len(k.weights)} entries but kpoints.points has {len(k.points)}",
            )
    if k.symprec <= 0:
        log.add("kpoints.symprec", f"must be positive, got {k.symprec}")


def _check_occupations(config: Config, log: ErrorLog) -> None:
    occ = config.occupations
    if occ.scheme == "methfessel-paxton" and occ.order < 1:
        log.add("occupations.order", f"must be at least 1, got {occ.order}")
    if occ.nbands is not None and occ.nbands < 1:
        log.add("occupations.nbands", f"must be positive, got {occ.nbands}")
    if occ.scheme == "tetrahedron" and config.kpoints.scheme == "gamma":
        log.add(
            "occupations.scheme",
            "'tetrahedron' needs a k-point mesh to tetrahedralise, but kpoints.scheme "
            "is 'gamma' (a single point). Use a Monkhorst-Pack grid or a smearing scheme",
        )


def _check_scf(config: Config, log: ErrorLog) -> None:
    scf = config.scf
    if scf.max_steps < 1:
        log.add("scf.max_steps", f"must be at least 1, got {scf.max_steps}")
    if not 0.0 < scf.mixing.beta <= 1.0:
        log.add("scf.mixing.beta", f"must lie in (0, 1], got {scf.mixing.beta}")
    if scf.mixing.history < 1:
        log.add("scf.mixing.history", f"must be at least 1, got {scf.mixing.history}")
    if scf.mixing.kerker_q0 <= 0 and scf.mixing.preconditioner == "kerker":
        log.add("scf.mixing.kerker_q0", f"must be positive, got {scf.mixing.kerker_q0}")
    for name, value in (("tol_energy", scf.tol_energy), ("tol_density", scf.tol_density)):
        if value <= 0:
            log.add(f"scf.{name}", f"must be positive, got {value}")
    if scf.method.startswith("direct") and config.occupations.scheme != "fixed":
        # Direct minimisation walks downhill on E(orbitals), which is only a smooth
        # function of the orbitals when the occupations are held fixed. With smearing the
        # occupations are variables too and the free energy has to be minimised over both,
        # which is a different algorithm. Catch it here rather than after the setup work.
        log.add(
            "scf.method",
            f"{scf.method!r} minimises the energy over the orbitals at fixed occupations, "
            f"but occupations.scheme is {config.occupations.scheme!r}. Use "
            "occupations.scheme: fixed for an insulator, or scf.method: mixing for a metal",
        )


def _check_solver_requirements(config: Config, log: ErrorLog) -> None:
    solver = config.solver
    if solver in _TWO_ENDPOINT_SOLVERS:
        for name in ("initial", "final"):
            if getattr(config.neb, name) is None:
                log.add(f"neb.{name}", f"is required when solver is {solver!r}")
        if config.neb.n_images < 3:
            log.add(
                "neb.n_images",
                f"needs at least 3 images (two endpoints and one intermediate), "
                f"got {config.neb.n_images}",
            )
    if solver == "relax" and config.relax.fmax <= 0:
        log.add("relax.fmax", f"must be positive, got {config.relax.fmax}")
    if solver == "md":
        if config.md.timestep <= 0:
            log.add("md.timestep", f"must be positive, got {config.md.timestep}")
        if config.md.steps < 1:
            log.add("md.steps", f"must be at least 1, got {config.md.steps}")
        if config.md.temperature < 0:
            log.add("md.temperature", f"must be non-negative, got {config.md.temperature}")
        if config.md.equilibration_steps >= config.md.steps:
            log.add(
                "md.equilibration_steps",
                f"({config.md.equilibration_steps}) leaves no production steps out of "
                f"md.steps ({config.md.steps})",
            )
        if config.md.ensemble == "nve" and config.md.temperature <= 0:
            # NVE still needs a temperature: it is what the initial Maxwell-Boltzmann
            # velocities are drawn from. It is simply not controlled afterwards -- the
            # run conserves energy, and the temperature settles near half its starting
            # value once the kinetic energy equipartitions with the potential.
            log.add(
                "md.temperature",
                "must be positive: it sets the initial Maxwell-Boltzmann velocities even "
                "in the 'nve' ensemble, where nothing thermostats it afterwards",
            )
    if solver == "dfpt" and not config.dfpt.perturbations:
        log.add("dfpt.perturbations", "must list at least one of: efield, phonon, strain")
    if solver == "phonon":
        _check_positive_int_triplet(
            config.phonon.supercell, "phonon.supercell", "phonon.supercell", log
        )
        if config.phonon.method == "finite-displacement" and config.phonon.displacement <= 0:
            log.add("phonon.displacement", f"must be positive, got {config.phonon.displacement}")
    if solver == "dos":
        _check_positive_int_triplet(config.dos.grid, "dos.grid", "dos.grid", log)
        if (
            config.dos.emin is not None
            and config.dos.emax is not None
            and config.dos.emin >= config.dos.emax
        ):
            log.add("dos.emax", f"must exceed dos.emin ({config.dos.emin})")
    if solver == "wannier" and config.wannier.backend == "native" and config.wannier.nwann is None:
        log.add("wannier.nwann", "is required for the native MLWF backend")


def _check_physics_consistency(config: Config, log: ErrorLog) -> None:
    """Combinations that parse but are physically inconsistent or unsupported."""
    hubbard = config.system.hubbard
    if hubbard.scheme != "none" and not hubbard.species:
        log.add(
            "system.hubbard.species",
            f"is empty but system.hubbard.scheme is {hubbard.scheme!r}; list the species "
            "the correction applies to, e.g. {Ni: {U: 6.0, J: 1.0}}",
        )
    if hubbard.scheme == "none" and hubbard.species:
        log.add(
            "system.hubbard.scheme",
            "is 'none', so the species entries would be silently ignored; set it to "
            "'dudarev' or 'liechtenstein'",
        )
    if hubbard.scheme == "dudarev":
        for symbol, params in hubbard.species.items():
            if params.U - params.J < 0:
                log.add(
                    f"system.hubbard.species.{symbol}",
                    f"gives U_eff = U - J = {params.U - params.J:g} eV < 0 under the "
                    "Dudarev scheme, which localises the shell in the wrong direction",
                )
    if hubbard.projector == "paw" and config.pseudo.method != "paw":
        log.add(
            "system.hubbard.projector",
            f"is 'paw' but pseudo.method is {config.pseudo.method!r}; PAW partial waves "
            "are only available with a PAW dataset. Use projector 'atomic' or 'wannier'",
        )

    if config.basis.type == "planewave" and config.basis.atomic_orbitals:
        log.add(
            "basis.atomic_orbitals",
            "only applies to basis.type 'mixed'; it would be ignored for 'planewave'",
        )

    # PBEsol PAW datasets exist only in UPF form (pslibrary); the PAW-XML tables
    # (JTH, GPAW) are PBE/LDA only. Flag it here rather than at dataset-load time.
    if config.xc.functional == "PBEsol" and config.pseudo.method == "paw":
        family = (config.pseudo.family or "").lower()
        if family and "psl" not in family and "pslibrary" not in family:
            log.add(
                "pseudo.family",
                f"{config.pseudo.family!r} is requested with xc PBEsol and method paw, but "
                "the PAW-XML tables (JTH, GPAW) ship PBE and LDA only. PBEsol PAW datasets "
                "exist only as pslibrary UPF files ('*.pbesol-*-kjpaw_psl.*')",
            )

    if config.parallel.band_groups < 1:
        log.add("parallel.band_groups", f"must be at least 1, got {config.parallel.band_groups}")
    if isinstance(config.parallel.kpoint_groups, int) and config.parallel.kpoint_groups < 1:
        log.add("parallel.kpoint_groups", "must be a positive integer or 'auto'")
    if config.output.checkpoint_every < 0:
        log.add("output.checkpoint_every", "must be non-negative (0 disables checkpointing)")


def _not_yet(log: ErrorLog, path: str, what: str, lands: str) -> None:
    log.add(path, f"{what} is not implemented yet -- it lands in {lands}")


def _check_not_implemented(config: Config, log: ErrorLog) -> None:
    """Settings the schema accepts but no code path reads yet.

    A setting that validates and is then ignored is the worst kind of input error: the run
    finishes, the output looks normal, and the result is not what was asked for (a
    ``system.hubbard`` block used to produce a plain DFT run). Such settings are refused
    here, as unimplemented solvers are, until the code that honours them lands. Only
    values that differ from what the code actually does are refused, so the defaults
    always pass.
    """
    if config.system.hubbard.scheme != "none":
        _not_yet(log, "system.hubbard.scheme", "DFT+U", "Phase 5 (analysis solvers)")
    if config.basis.type == "mixed":
        _not_yet(
            log,
            "basis.type",
            "the mixed plane-wave + atomic-orbital basis",
            "Phase 8, which is on hold (ADR-0015)",
        )
    if config.basis.type == "mixed" and config.basis.atomic_orbitals:
        _not_yet(log, "basis.atomic_orbitals", "the mixed basis", "Phase 8 (on hold)")
    if config.occupations.scheme == "tetrahedron":
        _not_yet(
            log,
            "occupations.scheme",
            "the linear tetrahedron method",
            "Phase 5 with the DOS solver; use a smearing scheme or 'fixed' meanwhile",
        )
    if config.scf.eigensolver != "davidson":
        _not_yet(
            log,
            "scf.eigensolver",
            f"{config.scf.eigensolver!r}",
            "Phase 9 (performance); use 'davidson'",
        )
    if config.scf.method.startswith("direct") and config.pseudo.method == "paw":
        log.add(
            "scf.method",
            f"{config.scf.method!r} is not implemented for PAW: direct minimisation keeps the "
            "orbitals orthonormal in the Euclidean metric, while PAW needs the S metric. Use "
            "scf.method: mixing for a PAW calculation",
        )
    parallel = config.parallel
    if parallel.numba:
        _not_yet(log, "parallel.numba", "numba-compiled kernels", "Phase 9 (performance)")
    if isinstance(parallel.kpoint_groups, int) and parallel.kpoint_groups > 1:
        _not_yet(
            log,
            "parallel.kpoint_groups",
            "MPI parallelism over k-points",
            "the Phase 5 period (ADR-0015); leave it at 'auto' or 1",
        )
    if parallel.band_groups > 1:
        _not_yet(log, "parallel.band_groups", "MPI parallelism over bands", "Phase 9")
    if config.output.checkpoint_every > 0:
        _not_yet(
            log,
            "output.checkpoint_every",
            "periodic checkpointing",
            "the native relaxation and MD solvers (ADR-0017); set it to 0. A finished SCF "
            "run already writes a results file that --restart can resume from",
        )


def resolve_and_validate(config: Config, log: ErrorLog) -> Config:
    """Run every cross-field check, recording problems in ``log``.

    Returns the config unchanged: derived defaults that depend on the pseudopotential
    datasets (recommended cutoffs, band count from the electron count) are filled in
    later, once the datasets have been loaded.
    """
    _check_energies(config, log)
    _check_kpoints(config, log)
    _check_occupations(config, log)
    _check_scf(config, log)
    _check_solver_requirements(config, log)
    _check_physics_consistency(config, log)
    _check_not_implemented(config, log)
    return config
