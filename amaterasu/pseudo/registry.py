"""Named pseudopotential families: where the files live, and what they promise.

amaterasu does not bundle pseudopotentials. Their licences differ from this project's
(CC BY 4.0 for PseudoDojo, GPL-3.0 for JTH), and bundling would propagate those terms.
Instead a family is named in the input, its manifest is fetched once and cached, and the
individual datasets are downloaded on demand and checked against the manifest's MD5.

The manifest is worth more than a file listing. PseudoDojo's ``.djson`` also carries, per
element:

* the exact basename, which is how ``Fe`` resolves to ``Fe-sp.psp8`` without guessing that
  the semicore variant is the recommended one;
* ``hints``, the generator's own recommended plane-wave cutoffs, which is what
  ``basis.ecut: null`` resolves to;
* ``dfact_meV``, the published delta-gauge deviation from the all-electron reference --
  the validation target amaterasu's own results are measured against later.

Only the network-facing helpers need a connection; everything else works against files
already on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..units import parse_energy
from .base import NCPseudopotential
from .readers import read_pseudopotential

__all__ = [
    "KNOWN_FAMILIES",
    "PseudoFamily",
    "PseudoResolutionError",
    "default_family",
    "fetch_family",
    "load_manifest",
    "load_pseudopotentials",
    "recommended_cutoff",
    "resolve_paths",
]

_JTH_RAW = "https://raw.githubusercontent.com/abinit/paw_jth_datasets/master"
_JTH_CITATION = (
    "F. Jollet, M. Torrent and N. Holzwarth, Comput. Phys. Commun. 185, 1246 (2014); "
    "JTH table v2.0, https://www.abinit.org/atomic_data/paw/ (GPL-3.0)"
)
_PSEUDODOJO_RAW = "https://raw.githubusercontent.com/abinit/pseudo_dojo/master/pseudo_dojo/pseudos"

_PSEUDODOJO_CITATION = (
    "M. J. van Setten et al., Comput. Phys. Commun. 226, 39 (2018); files CC BY 4.0"
)


class PseudoResolutionError(Exception):
    """Raised when the pseudopotentials a calculation needs cannot be assembled."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(
            "\n".join([f"{len(self.problems)} problem(s):", *(f"  {p}" for p in self.problems)])
        )


@dataclass(frozen=True)
class PseudoFamily:
    """One named table of pseudopotentials."""

    name: str
    #: ``nc`` or ``paw``; must match ``pseudo.method``.
    kind: str
    #: The functional the datasets were generated with, as an amaterasu functional name.
    xc: str
    file_format: str
    #: Directory within the upstream repository.
    directory: str
    #: The manifest inside it, listing basenames, checksums and cutoff hints. ``None`` for
    #: a table that publishes no manifest -- the JTH PAW datasets, whose file names follow
    #: from the element and whose recommended cutoffs are inside the files themselves.
    manifest: str | None = None
    #: File name template for a manifest-less family, formatted with ``element``.
    filename: str | None = None
    base_url: str = _PSEUDODOJO_RAW
    license: str = "CC BY 4.0"
    citation: str = _PSEUDODOJO_CITATION
    #: Which of the manifest's cutoff hints to use when none is requested.
    default_hint: str = "normal"
    #: Whether this is the family amaterasu picks for its ``(xc, method)`` combination.
    is_default: bool = False
    notes: str = ""

    @property
    def has_manifest(self) -> bool:
        return self.manifest is not None

    @property
    def manifest_url(self) -> str:
        return f"{self.base_url}/{self.directory}/{self.manifest}"

    def basename(self, element: str) -> str:
        """The file name this family uses for ``element``."""
        if self.filename is None:  # pragma: no cover - manifest families use their own
            raise ValueError(f"{self.name} publishes a manifest; look the basename up there")
        return self.filename.format(element=element)

    def file_url(self, basename: str) -> str:
        element = basename.split(".")[0].split("-")[0]
        return f"{self.base_url}/{self.directory}/{element}/{basename}"


#: Families amaterasu knows how to fetch. PBEsol PAW is deliberately absent: it exists
#: only as pslibrary UPF files, which are not laid out for programmatic fetching, so those
#: are supplied by hand through ``pseudo.files``.
KNOWN_FAMILIES: dict[str, PseudoFamily] = {
    family.name: family
    for family in [
        PseudoFamily(
            name="jth-pbe-v2.0",
            kind="paw",
            xc="PBE",
            file_format="pawxml",
            directory="pseudos/JTH-PBE-v2.0",
            filename="{element}.GGA_PBE-JTH.xml",
            base_url=_JTH_RAW,
            is_default=True,
            license="GPL-3.0",
            citation=_JTH_CITATION,
            notes=(
                "PAW-XML v0.7. Recommended cutoffs come from each file's own pw_ecut "
                "element rather than from a manifest."
            ),
        ),
        PseudoFamily(
            name="pseudodojo-nc-sr-pbe-standard-v0.4",
            is_default=True,
            kind="nc",
            xc="PBE",
            file_format="psp8",
            directory="ONCVPSP-PBE-PDv0.4",
            manifest="standard.djson",
        ),
        PseudoFamily(
            name="pseudodojo-nc-sr-pbe-stringent-v0.4",
            kind="nc",
            xc="PBE",
            file_format="psp8",
            directory="ONCVPSP-PBE-PDv0.4",
            manifest="stringent.djson",
            notes="harder datasets with smaller delta factors; needs a higher cutoff",
        ),
        PseudoFamily(
            name="pseudodojo-nc-sr-pbesol-standard-v0.4",
            is_default=True,
            kind="nc",
            xc="PBEsol",
            file_format="psp8",
            directory="ONCVPSP-PBEsol-PDv0.4",
            manifest="standard.djson",
        ),
        PseudoFamily(
            name="pseudodojo-nc-sr-pbesol-stringent-v0.4",
            kind="nc",
            xc="PBEsol",
            file_format="psp8",
            directory="ONCVPSP-PBEsol-PDv0.4",
            manifest="stringent.djson",
        ),
        PseudoFamily(
            name="pseudodojo-nc-sr-lda-standard-v0.4",
            is_default=True,
            kind="nc",
            xc="LDA-PW",
            file_format="psp8",
            directory="ONCVPSP-PW-PDv0.4",
            manifest="standard.djson",
        ),
    ]
}


def default_family(xc: str, method: str) -> str | None:
    """The family amaterasu picks when the input names a functional but no family."""
    for family in KNOWN_FAMILIES.values():
        if family.xc == xc and family.kind == method and family.is_default:
            return family.name
    return None


def _cache_directory(directory: str | os.PathLike) -> Path:
    path = Path(directory).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(
    family: PseudoFamily, directory: str | os.PathLike, *, download: bool = True
) -> dict:
    """Return the family's per-element metadata, fetching and caching it if needed."""
    cache = _cache_directory(directory) / family.name / "manifest.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if not download:
        raise PseudoResolutionError(
            [
                f"no cached manifest for {family.name}; run "
                f"'amaterasu fetch-pseudos --family {family.name}'"
            ]
        )
    try:
        with urllib.request.urlopen(family.manifest_url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PseudoResolutionError(
            [f"could not fetch the manifest for {family.name} from {family.manifest_url}: {exc}"]
        ) from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _element_entries(manifest: dict) -> dict:
    return manifest.get("pseudos_metadata", manifest)


def recommended_cutoff(
    family: PseudoFamily, symbols, directory, *, accuracy: str | None = None, download: bool = True
) -> float:
    """Largest recommended wavefunction cutoff over ``symbols``, in Hartree.

    PseudoDojo quotes its hints in Hartree. The maximum over the species in the cell is
    the right combination rule: a cutoff that is converged for the hardest element is
    converged for all of them, whereas an average would under-converge that element.
    """
    manifest = load_manifest(family, directory, download=download)
    entries = _element_entries(manifest)
    level = accuracy or family.default_hint
    cutoffs = []
    for symbol in symbols:
        entry = entries.get(symbol)
        if entry is None:
            raise PseudoResolutionError([f"{family.name} has no dataset for {symbol}"])
        hints = entry.get("hints", {})
        if level not in hints:
            raise PseudoResolutionError(
                [f"{family.name} gives no '{level}' cutoff hint for {symbol}; have {sorted(hints)}"]
            )
        cutoffs.append(float(hints[level]["ecut"]))
    return max(cutoffs)


def resolve_paths(
    family: PseudoFamily, symbols, directory, *, download: bool = True
) -> dict[str, Path]:
    """Map each element symbol to its dataset file, downloading any that are missing."""
    if not family.has_manifest:
        return _resolve_flat(family, symbols, directory, download=download)
    manifest = load_manifest(family, directory, download=download)
    entries = _element_entries(manifest)
    root = _cache_directory(directory) / family.name
    paths: dict[str, Path] = {}
    problems: list[str] = []

    for symbol in symbols:
        entry = entries.get(symbol)
        if entry is None:
            problems.append(f"{family.name} has no dataset for {symbol}")
            continue
        basename = entry["basename"]
        target = root / basename
        if not target.exists():
            if not download:
                problems.append(f"{symbol}: {target} is missing (run 'amaterasu fetch-pseudos')")
                continue
            try:
                _download(family.file_url(basename), target, entry.get("md5"))
            except PseudoResolutionError as exc:
                problems.extend(exc.problems)
                continue
        paths[symbol] = target

    if problems:
        raise PseudoResolutionError(problems)
    return paths


def _download(url: str, target: Path, expected_md5: str | None) -> None:
    """Fetch one dataset and verify it against the manifest's checksum."""
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PseudoResolutionError([f"could not download {url}: {exc}"]) from exc

    if expected_md5:
        digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        if digest != expected_md5:
            raise PseudoResolutionError(
                [f"{url} has md5 {digest} but the manifest says {expected_md5}; refusing to use it"]
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _resolve_flat(
    family: PseudoFamily, symbols, directory, *, download: bool = True
) -> dict[str, Path]:
    """Locate the files of a family whose names follow from the element symbol alone.

    The JTH PAW table publishes no manifest, so there is no checksum to verify and no
    list of elements to enumerate; what it does have is a predictable layout and each
    file's own recommended cutoffs.
    """
    root = _cache_directory(directory) / family.name
    paths: dict[str, Path] = {}
    problems: list[str] = []
    for symbol in symbols:
        basename = family.basename(symbol)
        target = root / symbol / basename
        if not target.exists():
            if not download:
                problems.append(
                    f"{symbol}: {target} is missing; run 'amaterasu fetch-pseudos "
                    f"--family {family.name} --elements {symbol}'"
                )
                continue
            _download(family.file_url(basename), target, None)
        paths[symbol] = target
    if problems:
        raise PseudoResolutionError(problems)
    return paths


def fetch_family(family_name: str, directory, symbols=None) -> dict[str, Path]:
    """Download a whole family, or just ``symbols``, into ``directory``."""
    family = KNOWN_FAMILIES.get(family_name)
    if family is None:
        raise PseudoResolutionError(
            [f"unknown family {family_name!r}; known: {', '.join(sorted(KNOWN_FAMILIES))}"]
        )
    if not family.has_manifest:
        if not symbols:
            raise PseudoResolutionError(
                [
                    f"{family.name} publishes no manifest, so its elements cannot be "
                    "enumerated; name them with --elements"
                ]
            )
        return resolve_paths(family, list(symbols), directory)
    manifest = load_manifest(family, directory)
    wanted = list(symbols) if symbols else sorted(_element_entries(manifest))
    return resolve_paths(family, wanted, directory)


@dataclass
class ResolvedPseudos:
    """The datasets a calculation will use, with everything derived from them."""

    pseudopotentials: dict[str, NCPseudopotential]
    family: PseudoFamily | None = None
    #: Recommended wavefunction cutoff in Hartree, when the family supplies hints.
    recommended_ecut: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def total_valence_electrons(self) -> float:
        return sum(p.valence_charge for p in self.pseudopotentials.values())


def load_pseudopotentials(config, symbols, *, download: bool = True) -> ResolvedPseudos:
    """Assemble and validate the datasets for a calculation.

    Explicit ``pseudo.files`` entries win over the family. The exchange-correlation
    functional each dataset was generated with is checked against ``xc.functional``:
    running PBEsol on PBE datasets is a genuine inconsistency, not a rounding detail, so
    it is an error unless ``pseudo.allow_xc_mismatch`` is set.
    """
    directory = Path(config.pseudo.directory).expanduser()
    symbols = list(dict.fromkeys(symbols))
    problems: list[str] = []
    warnings: list[str] = []
    paths: dict[str, Path] = {}

    family = None
    if config.pseudo.family:
        family = KNOWN_FAMILIES.get(config.pseudo.family)
        if family is None:
            problems.append(
                f"unknown pseudo.family {config.pseudo.family!r}; known families: "
                f"{', '.join(sorted(KNOWN_FAMILIES))}"
            )
        elif family.kind != config.pseudo.method:
            problems.append(
                f"pseudo.family {family.name!r} is a {family.kind} table but "
                f"pseudo.method is {config.pseudo.method!r}"
            )

    explicit = {s: Path(f).expanduser() for s, f in config.pseudo.files.items()}
    for symbol, path in explicit.items():
        paths[symbol] = path if path.is_absolute() or path.exists() else directory / path

    missing = [s for s in symbols if s not in paths]
    if missing and family is not None and not problems:
        try:
            paths.update(resolve_paths(family, missing, directory, download=download))
        except PseudoResolutionError as exc:
            problems.extend(exc.problems)
    elif missing and family is None and not problems:
        problems.append(
            f"no pseudopotential for {', '.join(missing)}: set pseudo.family or list the "
            "files explicitly under pseudo.files"
        )

    if problems:
        raise PseudoResolutionError(problems)

    loaded: dict[str, NCPseudopotential] = {}
    for symbol in symbols:
        path = paths[symbol]
        if not path.exists():
            problems.append(f"{symbol}: no such file {path}")
            continue
        try:
            pseudo = read_pseudopotential(path)
        except (ValueError, NotImplementedError) as exc:
            problems.append(f"{symbol}: cannot read {path.name}: {exc}")
            continue

        if pseudo.symbol != symbol:
            problems.append(f"{symbol}: {path.name} is a dataset for {pseudo.symbol}, not {symbol}")
        if pseudo.kind.lower() != config.pseudo.method.lower():
            problems.append(
                f"{symbol}: {path.name} is a {pseudo.kind} dataset but pseudo.method is "
                f"{config.pseudo.method!r}"
            )
        problems.extend(f"{symbol}: {p}" for p in pseudo.check())

        message = (
            f"{symbol}: {path.name} was generated with "
            f"{pseudo.xc_functional or pseudo.xc_description}, but xc.functional is "
            f"{config.xc.functional}. Mixing them is inconsistent: the pseudopotential "
            "no longer reproduces the all-electron atom for the functional in use."
        )
        if pseudo.xc_functional != config.xc.functional:
            if config.pseudo.allow_xc_mismatch:
                warnings.append(message + " Proceeding because allow_xc_mismatch is set.")
            else:
                problems.append(message + " Set pseudo.allow_xc_mismatch to override.")
        loaded[symbol] = pseudo

    if problems:
        raise PseudoResolutionError(problems)

    recommended = None
    if family is not None and family.has_manifest:
        try:
            recommended = recommended_cutoff(family, symbols, directory, download=download)
        except PseudoResolutionError as exc:
            warnings.extend(exc.problems)
    if recommended is None:
        # A PAW-XML file carries its generator's own recommended cutoffs; take the
        # strictest of the species present, which is what a mixed cell needs.
        hints = [
            p.ecut_hints["high"]
            for p in loaded.values()
            if getattr(p, "ecut_hints", None) and "high" in p.ecut_hints
        ]
        if hints:
            recommended = max(hints)

    return ResolvedPseudos(
        pseudopotentials=loaded, family=family, recommended_ecut=recommended, warnings=warnings
    )


def resolve_cutoffs(config, resolved: ResolvedPseudos) -> tuple[float, float]:
    """Return ``(ecut, ecut_rho)`` in Hartree, filling in defaults.

    ``ecut_rho`` defaults to ``4 * ecut`` for norm-conserving datasets: the density is
    ``|psi|^2``, whose Fourier components reach twice the largest wavefunction G, and the
    energy goes as G^2.
    """
    if config.basis.ecut is not None:
        ecut = parse_energy(config.basis.ecut, "Ry")
    elif resolved.recommended_ecut is not None:
        ecut = resolved.recommended_ecut
    else:
        raise PseudoResolutionError(
            ["basis.ecut is not set and the datasets carry no recommended cutoff"]
        )
    if config.basis.ecut_rho is not None:
        ecut_rho = parse_energy(config.basis.ecut_rho, "Ry")
    else:
        ecut_rho = _default_density_cutoff(ecut, resolved)
    return ecut, ecut_rho


#: Ratio ``ecut_rho / ecut`` used when the input does not give one.
DENSITY_CUTOFF_RATIO = {"NC": 4.0, "PAW": 8.0}


def _default_density_cutoff(ecut: float, resolved: ResolvedPseudos) -> float:
    """The density cutoff to use when ``basis.ecut_rho`` is not set.

    For norm-conserving datasets the factor is four: the density is ``|psi|^2``, whose
    Fourier components reach twice the largest wavefunction ``G``, and the energy goes as
    ``G^2``. PAW needs more, because the sharpest object in its density is not the
    orbitals at all but the compensation charge, which has to fit inside the augmentation
    sphere. Eight is a starting value, not a converged one: it is the parameter every PAW
    code exposes separately (ABINIT calls it ``pawecutdg``), and it has to be converged like
    any other. :func:`amaterasu.hamiltonian.paw.build_paw_machinery` measures what the grid
    still misses of the compensation charge at the cutoff actually in use.
    """
    kinds = {getattr(p, "kind", "NC") for p in resolved.pseudopotentials.values()}
    return ecut * max(DENSITY_CUTOFF_RATIO.get(kind, 4.0) for kind in kinds)
