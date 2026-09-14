"""Reader for the UPF v2 pseudopotential format used by Quantum ESPRESSO and pslibrary.

UPF matters to amaterasu for one specific reason beyond breadth of support: **PBEsol PAW
datasets exist only in this format**. The PAW-XML tables (JTH, GPAW) cover PBE and LDA
only, so a PBEsol PAW calculation has to read pslibrary's ``*-kjpaw_psl`` UPF files.

This reader handles the norm-conserving content, which PAW and ultrasoft files also carry;
the augmentation data that makes a file PAW is parsed as far as recording that it is there,
and reading it properly is Phase 4 work.

Units and conventions, all verified numerically against real files rather than assumed:

``PP_LOCAL``
    **Rydberg**, approaching ``-2 Z_ion / r``. Converted to Hartree on load, which is the
    single most likely place for a factor of two to go unnoticed.
``PP_DIJ``
    Rydberg, converted likewise. Exactly diagonal for a norm-conserving dataset (it is
    then just the Kleinman-Bylander energies); a PAW dataset has off-diagonal couplings.
``PP_BETA.i``
    ``r beta_l(r)``, unit-normalised for norm-conserving datasets only.
``PP_RHOATOM``
    Already ``4 pi r^2 rho(r)``, matching amaterasu's internal convention, so it is stored
    unchanged and integrates directly to ``Z_ion``.
``PP_NLCC``
    A bare ``rho_c(r)``, finite at the origin, so it *is* rescaled by ``4 pi r^2``.

Some UPF files in the wild contain characters that are illegal in XML (bare ``&`` in the
free-text ``PP_INFO`` block, most often). The reader retries with that block stripped
rather than failing on a file every other code accepts.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ...units import RYDBERG_IN_HARTREE
from ..base import NCPseudopotential, Projector, RadialGrid

__all__ = ["read_upf"]

#: Quantum ESPRESSO's four-component functional string -> amaterasu functional name.
#: The components are (exchange, correlation, gradient exchange, gradient correlation).
_QE_FUNCTIONAL_COMPONENTS = {
    ("SLA", "PZ", "NOGX", "NOGC"): "LDA-PZ",
    ("SLA", "PW", "NOGX", "NOGC"): "LDA-PW",
    ("SLA", "PW", "PBX", "PBC"): "PBE",
    ("SLA", "PW", "PSX", "PSC"): "PBEsol",
}

#: Short names some generators write instead of the four-component string.
_SHORT_FUNCTIONAL_NAMES = {
    "PBE": "PBE",
    "PBESOL": "PBEsol",
    "PZ": "LDA-PZ",
    "LDA": "LDA-PZ",
    "PW": "LDA-PW",
    "PW91": None,
    "BLYP": None,
    "REVPBE": None,
    "RPBE": None,
}


def _decode_functional(raw: str) -> tuple[str | None, str]:
    """Map a UPF ``functional`` attribute to ``(amaterasu name, description)``."""
    text = (raw or "").strip()
    if not text:
        return None, "unspecified"
    components = tuple(text.upper().split())
    # A pure LDA is sometimes written with the two gradient slots left off entirely
    # rather than filled with NOGX/NOGC; both spellings mean the same functional.
    if len(components) == 2:
        components = (*components, "NOGX", "NOGC")
    if len(components) == 4:
        name = _QE_FUNCTIONAL_COMPONENTS.get(components)
        return name, f"QE {' '.join(components)}"
    if len(components) == 1:
        key = components[0]
        return _SHORT_FUNCTIONAL_NAMES.get(key), key
    return None, text


def _read_text(path: Path) -> str:
    """Read a file, transparently decompressing ``.gz`` (how GPAW ships its datasets)."""
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_xml(path: Path) -> ET.Element:
    """Parse a UPF file, tolerating the unescaped ``&`` that plague ``PP_INFO``."""
    text = _read_text(path)
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        pass
    stripped = re.sub(r"<PP_INFO>.*?</PP_INFO>", "<PP_INFO/>", text, flags=re.DOTALL)
    # Escape any remaining bare ampersands, which are the usual remaining offender.
    stripped = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", stripped)
    try:
        return ET.fromstring(stripped)
    except ET.ParseError as exc:
        raise ValueError(f"{path.name}: not parseable as UPF XML ({exc})") from exc


def _text_array(element: ET.Element | None, name: str, source: str) -> np.ndarray:
    if element is None or element.text is None:
        raise ValueError(f"{source}: required block {name} is missing or empty")
    return np.fromstring(element.text, sep=" ")


def _atomic_number(symbol: str, attributes: dict) -> float:
    """UPF's PP_HEADER carries no nuclear charge, so take it from the element symbol."""
    if "atomic_number" in attributes:
        return float(attributes["atomic_number"])
    from ase.data import atomic_numbers

    number = atomic_numbers.get(symbol)
    if number is None:
        raise ValueError(f"unknown chemical symbol {symbol!r} in PP_HEADER")
    return float(number)


def read_upf(path: str | os.PathLike) -> NCPseudopotential:
    """Read a UPF v2 file into an :class:`NCPseudopotential`.

    PAW and ultrasoft files are read for their norm-conserving content; the augmentation
    charges they also carry are noted in ``extra`` but not yet interpreted, so such a
    dataset is not yet usable for a calculation.
    """
    path = Path(path)
    source = path.name
    root = _parse_xml(path)
    if not root.tag.startswith("UPF"):
        raise ValueError(f"{source}: root element is <{root.tag}>, expected <UPF>")

    header = root.find("PP_HEADER")
    if header is None:
        raise ValueError(f"{source}: no PP_HEADER block")
    attributes = {k: (v.strip() if isinstance(v, str) else v) for k, v in header.attrib.items()}

    symbol = attributes.get("element", path.stem.split(".")[0]).strip()
    pseudo_type = attributes.get("pseudo_type", "NC").upper()
    valence_charge = float(attributes["z_valence"])
    l_max = int(attributes.get("l_max", 0))
    l_local = int(attributes.get("l_local", -1))
    n_projectors = int(attributes.get("number_of_proj", 0))

    if attributes.get("has_so", "F").upper() in ("T", "TRUE", ".TRUE."):
        raise NotImplementedError(
            f"{source}: this dataset carries spin-orbit projectors, which amaterasu does "
            "not read yet; use the scalar-relativistic variant"
        )

    mesh = root.find("PP_MESH")
    radii = _text_array(mesh.find("PP_R") if mesh is not None else None, "PP_MESH/PP_R", source)
    rab_element = mesh.find("PP_RAB") if mesh is not None else None
    dr_di = _text_array(rab_element, "PP_MESH/PP_RAB", source) if rab_element is not None else None
    # A UPF mesh is exponential unless the spacing happens to be uniform.
    uniform = np.allclose(np.diff(radii), radii[1] - radii[0], rtol=1e-9)
    grid = RadialGrid(
        r=np.ascontiguousarray(radii), kind="linear" if uniform else "log", dr_di=dr_di
    )

    local_potential = _text_array(root.find("PP_LOCAL"), "PP_LOCAL", source)
    local_potential = local_potential * RYDBERG_IN_HARTREE

    nonlocal_block = root.find("PP_NONLOCAL")
    projectors: list[Projector] = []
    dij = None
    if nonlocal_block is not None:
        betas = sorted(
            (e for e in nonlocal_block if e.tag.startswith("PP_BETA")),
            key=lambda e: int(e.tag.split(".")[-1]) if "." in e.tag else 0,
        )
        per_l_index: dict[int, int] = {}
        for element in betas:
            angular = int(element.attrib.get("angular_momentum", 0))
            values = np.asarray(element.text.split(), dtype=float)
            if len(values) < len(radii):
                # Some generators truncate beta at its cutoff radius; pad with zeros so
                # every radial array shares one grid.
                values = np.concatenate([values, np.zeros(len(radii) - len(values))])
            index = per_l_index.get(angular, 0)
            per_l_index[angular] = index + 1
            projectors.append(
                Projector(
                    l=angular,
                    index=index,
                    energy=0.0,  # filled in from D_ij below
                    values=np.ascontiguousarray(values[: len(radii)]),
                )
            )
        dij_element = nonlocal_block.find("PP_DIJ")
        if dij_element is not None and projectors:
            count = len(projectors)
            flat = _text_array(dij_element, "PP_DIJ", source) * RYDBERG_IN_HARTREE
            if flat.size != count * count:
                raise ValueError(
                    f"{source}: PP_DIJ holds {flat.size} numbers, expected "
                    f"{count}x{count} for {count} projectors"
                )
            dij = flat.reshape(count, count)
            projectors = [
                Projector(p.l, p.index, float(dij[i, i]), p.values)
                for i, p in enumerate(projectors)
            ]

    if n_projectors and len(projectors) != n_projectors:
        raise ValueError(
            f"{source}: PP_HEADER declares {n_projectors} projectors but "
            f"{len(projectors)} PP_BETA blocks are present"
        )

    # PP_RHOATOM is already 4 pi r^2 rho; PP_NLCC is a bare rho and needs the factor.
    rho_element = root.find("PP_RHOATOM")
    valence_density = (
        np.ascontiguousarray(_text_array(rho_element, "PP_RHOATOM", source))
        if rho_element is not None
        else None
    )
    nlcc_element = root.find("PP_NLCC")
    core_density = (
        np.ascontiguousarray(_text_array(nlcc_element, "PP_NLCC", source) * 4.0 * np.pi * radii**2)
        if nlcc_element is not None
        else None
    )

    functional, description = _decode_functional(attributes.get("functional", ""))

    return NCPseudopotential(
        symbol=symbol,
        atomic_number=_atomic_number(symbol, attributes),
        valence_charge=valence_charge,
        grid=grid,
        local_potential=np.ascontiguousarray(local_potential),
        projectors=projectors,
        l_max=l_max,
        l_local=l_local,
        xc_functional=functional,
        xc_description=description,
        core_density=core_density,
        valence_density=valence_density,
        kind="PAW" if pseudo_type == "PAW" else "NC",
        dij=dij,
        generator=attributes.get("generated", ""),
        source=str(path),
        extra={
            "format": "UPF",
            "version": root.attrib.get("version", "2"),
            "pseudo_type": pseudo_type,
            "is_ultrasoft": attributes.get("is_ultrasoft", "F"),
            "functional_raw": attributes.get("functional", ""),
            "has_augmentation": root.find(".//PP_AUGMENTATION") is not None,
        },
    )
