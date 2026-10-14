"""Command line interface: ``amaterasu <command> ...``.

Commands
--------
``run``            run a calculation described by a YAML input file
``info``           summarise a structure file, or validate a calculation input
``convert``        convert a structure between the amaterasu format and ASE formats
``fetch-pseudos``  download a pseudopotential / PAW family        (Phase 1)
``convergence``    scan one input parameter to check convergence  (Phase 2)
``plot``           plot bands, DOS or phonons from a result file  (Phase 5)

A calculation is driven by two files: a plain-text structure file (lattice vectors,
species, coordinates) and a YAML file holding the numerical settings. Splitting them means
a parameter sweep varies one file while the other is held fixed, which is what
``--set path.to.key=value`` is for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, config_to_yaml, load_config
from .io.structure_read import StructureParseError, read_structure
from .io.structure_write import write_amaterasu_structure
from .pseudo.registry import (
    KNOWN_FAMILIES,
    PseudoResolutionError,
    fetch_family,
    load_pseudopotentials,
    resolve_cutoffs,
)
from .solvers import bands as bands_solver
from .solvers import md as md_solver
from .solvers import relax as relax_solver
from .solvers import scf as scf_solver
from .system.kpoints import generate as generate_kpoints
from .system.structure import summarize
from .system.symmetry import symprec_scan
from .units import HARTREE_IN_EV, OUTPUT_UNITS

__all__ = ["main"]

_UNIT_BANNER = (
    "output units: energy {energy}, length {length}, force {force}, "
    "stress {stress} in files (positive under compression), pressure {pressure}, "
    "MD time {time}"
).format(**OUTPUT_UNITS)

#: Solvers whose implementation phase has not landed yet, with the phase that will add them.
_UNIMPLEMENTED_SOLVERS = {
    "neb": "Phase 5 (analysis solvers)",
    "dos": "Phase 5 (analysis solvers)",
    "phonon": "Phase 5 (analysis solvers)",
    "dfpt": "Phase 6 (density functional perturbation theory)",
    "wannier": "Phase 7 (Wannier functions)",
}


def _resolve_structure_path(config: Config, input_path: Path, override: str | None) -> Path:
    """Locate the structure file; paths in the YAML are relative to the YAML itself."""
    if override is not None:
        return Path(override)
    candidate = Path(config.structure)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return input_path.parent / candidate


def _describe_run(config: Config, structure_path: Path, atoms) -> str:
    """The header block printed before a run (and the whole output of ``--dry-run``)."""
    summary = summarize(atoms, symprec=config.kpoints.symprec)
    kpoints = generate_kpoints(atoms, config.kpoints)
    ecut = config.basis.ecut if config.basis.ecut is not None else "from dataset"
    lines = [
        f"amaterasu {__version__}",
        _UNIT_BANNER,
        "",
        f"solver               {config.solver}",
        f"functional           {config.xc.functional}",
        f"pseudopotentials     {config.pseudo.method}"
        + (f"  family={config.pseudo.family}" if config.pseudo.family else ""),
        f"basis                {config.basis.type}   ecut={ecut}",
        f"spin                 {config.system.spin}",
        f"k-points             {kpoints}",
        f"occupations          {config.occupations.scheme}  smearing={config.occupations.smearing}",
        f"scf                  {config.scf.method}"
        + (
            f"  ({config.scf.mixing.algorithm} + {config.scf.mixing.preconditioner})"
            if config.scf.method == "mixing"
            else f"  (history={config.scf.lbfgs_history})"
        ),
        f"  eigensolver        {config.scf.eigensolver}",
        "",
        f"structure            {structure_path}",
        summary.render(),
    ]
    if config.system.hubbard.scheme != "none":
        species = ", ".join(
            f"{s}: U={p.U} eV J={p.J} eV" for s, p in config.system.hubbard.species.items()
        )
        lines.append(
            f"Hubbard              {config.system.hubbard.scheme} "
            f"(projector: {config.system.hubbard.projector})   {species}"
        )
        lines.append(
            "  note: a U value is only meaningful with the projector it was fitted for; "
            "the projector is recorded in the output."
        )
    return "\n".join(lines)


def _check_restart(config: Config, restart: str | None) -> None:
    """Refuse a ``--restart`` that the chosen solver would silently ignore."""
    if restart is None:
        return
    if config.solver != "scf":
        raise ConfigError(
            f"--restart is only implemented for solver 'scf', but the input asks for "
            f"{config.solver!r}; it would be ignored. Restarting relaxations and MD lands "
            "with the native solvers (ADR-0017)"
        )
    if config.scf.method != "mixing":
        raise ConfigError(
            f"--restart needs scf.method: mixing; {config.scf.method!r} always starts from "
            "fresh orbitals, and restarting it from a stored density is not implemented"
        )


def _command_run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    config = load_config(input_path, overrides=args.set)
    if args.output_dir is not None:
        config.output.directory = args.output_dir
    _check_restart(config, args.restart)
    say = _make_logger(config.output.log_level)

    structure_path = _resolve_structure_path(config, input_path, args.structure)
    atoms = read_structure(structure_path)
    # A dry run exists to show the resolved settings, so it ignores output.log_level.
    (print if args.dry_run else say)(_describe_run(config, structure_path, atoms))

    if args.dry_run:
        print("\n--- resolved input (all defaults expanded) ---")
        print(config_to_yaml(config), end="")
        print("--- dry run: no calculation performed ---")
        return 0

    outdir = Path(config.output.directory).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    resolved = outdir / "resolved_input.yaml"
    resolved.write_text(config_to_yaml(config), encoding="utf-8")
    say(f"\nwrote {resolved}")

    if config.solver == "scf":
        say()
        run = scf_solver.run(
            atoms,
            config,
            output_directory=outdir,
            logger=say,
            restart=args.restart,
        )
        return 0 if run.result.converged else 1

    if config.solver == "relax":
        say()
        relaxation = relax_solver.run(atoms, config, output_directory=outdir, logger=say)
        say(relaxation.render())
        say(f"\nwrote {relaxation.trajectory}")
        return 0 if relaxation.converged else 1

    if config.solver == "bands":
        say()
        structure = bands_solver.run(atoms, config, output_directory=outdir, logger=say)
        return 0 if structure.eigenvalues.size else 1

    if config.solver == "md":
        say()
        trajectory = md_solver.run(atoms, config, output_directory=outdir, logger=say)
        say(trajectory.render(len(atoms)))
        say(f"\nwrote {trajectory.trajectory}")
        return 0

    phase = _UNIMPLEMENTED_SOLVERS.get(config.solver, "a later phase")
    print(
        f"\nsolver '{config.solver}' is not implemented yet -- it lands in {phase}.\n"
        "The input above validated cleanly, so it will run unchanged once that phase is in."
    )
    return 3


def _report_pseudopotentials(config, atoms, *, download: bool) -> int:
    """Resolve and describe the datasets a calculation needs. Returns an exit code."""
    symbols = sorted(set(atoms.get_chemical_symbols()))
    print(f"\npseudopotentials for {', '.join(symbols)}")
    print(f"  directory          {Path(config.pseudo.directory).expanduser()}")
    try:
        resolved = load_pseudopotentials(config, symbols, download=download)
    except PseudoResolutionError as exc:
        for problem in exc.problems:
            print(f"  ERROR: {problem}", file=sys.stderr)
        return 2

    if resolved.family is not None:
        print(f"  family             {resolved.family.name}  ({resolved.family.license})")
    for symbol in symbols:
        print(f"  {symbol:<3} {resolved.pseudopotentials[symbol]}")
    print(f"  valence electrons  {resolved.total_valence_electrons:g}")

    ecut, ecut_rho = resolve_cutoffs(config, resolved)
    origin = "from input" if config.basis.ecut is not None else "recommended by the dataset"
    print(f"  ecut               {ecut:g} Ha = {2 * ecut:g} Ry   ({origin})")
    print(f"  ecut_rho           {ecut_rho:g} Ha = {2 * ecut_rho:g} Ry")
    if resolved.recommended_ecut is not None and ecut < resolved.recommended_ecut:
        print(
            f"  WARNING: the dataset recommends at least {resolved.recommended_ecut:g} Ha "
            f"({2 * resolved.recommended_ecut:g} Ry); {ecut:g} Ha may be under-converged"
        )
    for warning in resolved.warnings:
        print(f"  WARNING: {warning}")
    return 0


def _command_info(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.suffix.lower() in (".yaml", ".yml"):
        config = load_config(path, overrides=args.set)
        structure_path = _resolve_structure_path(config, path, args.structure)
        atoms = read_structure(structure_path)
        print(_describe_run(config, structure_path, atoms))
        return _report_pseudopotentials(config, atoms, download=not args.offline)

    atoms = read_structure(path)
    print(f"amaterasu {__version__}")
    print(_UNIT_BANNER)
    print()
    print(summarize(atoms, symprec=args.symprec).render(symprec_sensitivity=symprec_scan(atoms)))
    return 0


def _command_convert(args: argparse.Namespace) -> int:
    atoms = read_structure(args.input, format=args.from_format)
    output = Path(args.output)
    if output.suffix.lower() in (".str", ".amt", "") or args.to_format == "amaterasu":
        write_amaterasu_structure(atoms, output, comment=f"converted from {args.input}")
    else:
        from ase.io import write as ase_write

        ase_write(output, atoms, format=args.to_format)
    print(f"wrote {output}  ({atoms.get_chemical_formula()}, {len(atoms)} atoms)")
    return 0


def _command_families(args: argparse.Namespace) -> int:
    print(f"amaterasu {__version__}\n")
    print(f"{'family':<40} {'kind':<5} {'functional':<10} {'format':<6} licence")
    for name in sorted(KNOWN_FAMILIES):
        family = KNOWN_FAMILIES[name]
        print(
            f"{name:<40} {family.kind:<5} {family.xc:<10} {family.file_format:<6} {family.license}"
        )
        if family.notes:
            print(f"{'':<40} {family.notes}")
    print(
        "\nPBEsol PAW datasets are not listed: they exist only as pslibrary UPF files "
        "(*-kjpaw_psl),\nwhich are not laid out for programmatic fetching. Download one "
        "and point pseudo.files at\nit -- amaterasu reads the Kresse-Joubert formulation "
        "those files are written in."
    )
    return 0


def _command_fetch_pseudos(args: argparse.Namespace) -> int:
    symbols = [s.strip() for s in args.elements.split(",")] if args.elements else None
    try:
        paths = fetch_family(args.family, args.directory, symbols)
    except PseudoResolutionError as exc:
        for problem in exc.problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2
    directory = Path(args.directory).expanduser()
    print(f"fetched {len(paths)} dataset(s) into {directory / args.family}")
    for symbol in sorted(paths):
        print(f"  {symbol:<3} {paths[symbol].name}")
    family = KNOWN_FAMILIES[args.family]
    print(f"\nlicence: {family.license}\nplease cite: {family.citation}")
    return 0


def _flushing_print(*args) -> None:
    """Print and flush, so a long run is followed live through a redirected log."""
    print(*args, flush=True)


#: Line prefixes that mark a message as a warning for ``output.log_level: warning``.
_WARNING_PREFIXES = ("WARNING", "NOTE")


def _make_logger(level: str):
    """The run log for ``output.log_level``.

    ``debug`` and ``info`` print everything; ``warning`` prints only the lines that start
    with WARNING or NOTE (after leading whitespace); ``error`` prints nothing, because
    errors go to stderr through the exception handler in :func:`main`. The results are in
    the output files whatever the level.
    """
    if level in ("debug", "info"):
        return _flushing_print
    if level == "warning":

        def warnings_only(*args) -> None:
            text = " ".join(str(a) for a in args)
            if text.lstrip().startswith(_WARNING_PREFIXES):
                _flushing_print(*args)

        return warnings_only
    return lambda *_: None


def _parse_scan_range(text: str) -> list[float]:
    """Parse ``START:STOP:STEP`` into an inclusive list of values."""
    parts = text.split(":")
    if len(parts) != 3:
        raise ConfigError(f"--range must be START:STOP:STEP, got {text!r}")
    try:
        start, stop, step = (float(x) for x in parts)
    except ValueError as exc:
        raise ConfigError(f"--range must be three numbers, got {text!r}") from exc
    if step <= 0:
        raise ConfigError(f"--range step must be positive, got {step}")
    values = []
    current = start
    while current <= stop + 1e-9 * max(1.0, abs(stop)):
        values.append(round(current, 10))
        current += step
    if len(values) < 2:
        raise ConfigError(f"--range {text!r} yields fewer than two points")
    return values


def _scan_override(path: str, value: float) -> str:
    """Format one ``--set`` override for the scanned parameter.

    ``kpoints.grid`` is special-cased: a scan over it means an ``n x n x n`` mesh, which
    is what a k-point convergence test actually varies.
    """
    if path == "kpoints.grid":
        n = round(value)
        return f"kpoints.grid=[{n},{n},{n}]"
    formatted = int(value) if float(value).is_integer() else value
    return f"{path}={formatted}"


def _command_convergence(args: argparse.Namespace) -> int:
    """Scan one input parameter and report how the total energy settles.

    Convergence must be established one parameter at a time: the cutoff with the k-point
    mesh held fixed, then the mesh with the cutoff held fixed. Scanning both together
    hides which one is responsible, and the two errors can cancel at a particular pair of
    values and reappear elsewhere. The header therefore states what is being held fixed.
    """
    input_path = Path(args.input)
    values = _parse_scan_range(args.range)
    base = load_config(input_path, overrides=args.set)
    if base.solver != "scf":
        print(
            f"convergence scans run the 'scf' solver; the input asks for {base.solver!r}",
            file=sys.stderr,
        )
        return 2

    structure_path = _resolve_structure_path(base, input_path, args.structure)
    atoms = read_structure(structure_path)
    print(f"amaterasu {__version__}")
    print(_UNIT_BANNER)
    print(f"\nconvergence scan of {args.scan} over {values}")
    print(f"structure            {structure_path}")
    print(f"functional           {base.xc.functional}")
    held = {
        "basis.ecut": f"kpoints {base.kpoints.scheme} {base.kpoints.grid}",
        "kpoints.grid": f"basis.ecut {base.basis.ecut}",
    }.get(args.scan, "everything else at the input values")
    print(f"held fixed           {held}")
    print(
        "\nA scan is only meaningful with everything else fixed: converge the cutoff at a\n"
        "fixed mesh, then the mesh at a fixed cutoff, and never both at once.\n"
    )

    natoms = len(atoms)
    rows = []
    print(f"  {args.scan:>14s} {'E (eV/atom)':>18s} {'dE (meV/atom)':>15s} {'iters':>6s} {'s':>7s}")
    for value in values:
        config = load_config(input_path, overrides=[*args.set, _scan_override(args.scan, value)])
        config.output.directory = args.output_dir or config.output.directory
        try:
            run = scf_solver.run(
                atoms,
                config,
                logger=lambda *_: None,
                write_output=False,
                download=not args.offline,
                compute_properties=False,
            )
        except PseudoResolutionError as exc:
            for problem in exc.problems:
                print(f"  ERROR: {problem}", file=sys.stderr)
            return 2
        energy = run.result.energies.extrapolated * HARTREE_IN_EV / natoms
        change = (energy - rows[-1][1]) * 1000 if rows else float("nan")
        rows.append((value, energy, run.result.iterations, run.result.seconds))
        print(
            f"  {value:14g} {energy:18.9f} {change:15.3f} "
            f"{run.result.iterations:6d} {run.result.seconds:7.1f}",
            flush=True,
        )

    reference = rows[-1][1]
    print(f"\n  differences from the finest point ({args.scan} = {rows[-1][0]:g}):")
    for value, energy, _, _ in rows:
        print(f"    {value:14g} {(energy - reference) * 1000:12.3f} meV/atom")
    threshold = args.threshold
    converged = [row for row in rows if abs(row[1] - reference) * 1000 < threshold]
    if converged:
        print(
            f"\n  within {threshold:g} meV/atom of the finest point from "
            f"{args.scan} = {converged[0][0]:g} onwards"
        )
    else:
        print(f"\n  no point is within {threshold:g} meV/atom; extend the range")
    return 0


def _command_not_yet(name: str, phase: str):
    def handler(args: argparse.Namespace) -> int:
        print(f"'amaterasu {name}' is not implemented yet -- it lands in {phase}.", file=sys.stderr)
        return 3

    return handler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amaterasu",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_UNIT_BANNER,
    )
    parser.add_argument("--version", action="version", version=f"amaterasu {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    run = subparsers.add_parser("run", help="run a calculation from a YAML input file")
    run.add_argument("input", help="calculation input file (YAML)")
    run.add_argument("--structure", help="structure file, overriding the one named in the YAML")
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override one setting, e.g. --set kpoints.grid='[4,4,4]' (repeatable)",
    )
    run.add_argument("-o", "--output-dir", help="output directory, overriding output.directory")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the input and print the fully resolved settings without computing",
    )
    run.add_argument("--restart", help="resume from an HDF5 checkpoint")
    run.set_defaults(func=_command_run)

    info = subparsers.add_parser(
        "info", help="summarise a structure file, or validate a calculation input"
    )
    info.add_argument("path", help="structure file, or a .yaml calculation input")
    info.add_argument("--structure", help="structure file, when PATH is a YAML input")
    info.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help="override one setting"
    )
    info.add_argument(
        "--symprec", type=float, default=1e-5, help="symmetry tolerance in Angstrom (default 1e-5)"
    )
    info.add_argument(
        "--offline",
        action="store_true",
        help="do not download missing pseudopotentials; report them as missing instead",
    )
    info.set_defaults(func=_command_info)

    convert = subparsers.add_parser("convert", help="convert a structure between formats")
    convert.add_argument("input", help="input structure (amaterasu format, POSCAR, CIF, xyz, ...)")
    convert.add_argument("-o", "--output", required=True, help="output file")
    convert.add_argument("--from-format", help="force the input format instead of auto-detecting")
    convert.add_argument("--to-format", help="force the output format (default: from the suffix)")
    convert.set_defaults(func=_command_convert)

    fetch = subparsers.add_parser("fetch-pseudos", help="download a pseudopotential family")
    fetch.add_argument(
        "--family",
        required=True,
        help="family name; use --list to see what is available",
    )
    fetch.add_argument("--elements", help="comma-separated symbols; default is the whole table")
    fetch.add_argument(
        "-d", "--directory", default="~/.amaterasu/pseudos", help="where to store the files"
    )
    fetch.set_defaults(func=_command_fetch_pseudos)

    families = subparsers.add_parser("families", help="list the known pseudopotential families")
    families.set_defaults(func=_command_families)

    convergence = subparsers.add_parser("convergence", help="scan one parameter for convergence")
    convergence.add_argument("input")
    convergence.add_argument("--scan", required=True, metavar="PATH", help="e.g. basis.ecut")
    convergence.add_argument("--range", required=True, metavar="START:STOP:STEP")
    convergence.add_argument("--structure", help="structure file, overriding the YAML")
    convergence.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help="override one setting"
    )
    convergence.add_argument("-o", "--output-dir", help="output directory")
    convergence.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="convergence criterion in meV/atom against the finest point (default 1)",
    )
    convergence.add_argument(
        "--offline", action="store_true", help="do not download missing pseudopotentials"
    )
    convergence.set_defaults(func=_command_convergence)

    plot = subparsers.add_parser("plot", help="plot results from an output file")
    plot.add_argument("kind", choices=["bands", "dos", "phonon"])
    plot.add_argument("results")
    plot.add_argument("-o", "--output", default=None)
    plot.set_defaults(func=_command_not_yet("plot", "Phase 5 (analysis solvers)"))

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling ``sys.exit``."""
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, StructureParseError, PseudoResolutionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
