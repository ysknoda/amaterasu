"""Reader for the PAW-XML v0.7 atomic dataset format.

PAW-XML is the format of the ABINIT/JTH and GPAW tables. It is a published specification
with named elements and Hartree atomic units throughout, which makes it the easiest of the
PAW formats to read correctly -- the awkward parts are the radial grid definitions, of
which the specification allows six, and the fact that different functions in one file may
live on different grids.

The conventions this reader relies on were confirmed numerically on a real file rather
than taken on trust; see :mod:`amaterasu.pseudo.paw.dataset` for the list. The single
sharpest confirmation is that ``integral p~_i phi~_j r^2 dr`` comes out as the identity to
1e-14, which would fail by orders of magnitude under the ``r f(r)`` convention that UPF
uses for the same objects.

Elements that are read
----------------------
``atom``, ``xc_functional``, ``generator``, ``paw_radius``, ``pw_ecut``, ``valence_states``,
``radial_grid``, ``ae_partial_wave``, ``pseudo_partial_wave``, ``projector_function``,
``ae_core_density``, ``pseudo_core_density``, ``pseudo_valence_density``,
``zero_potential``, ``kinetic_energy_differences``.

``shape_function`` is read for provenance only. The compensation-charge shape is a choice
made by the code rather than by the dataset -- it cancels out of the total energy -- and
amaterasu uses its own Gaussian; see :mod:`amaterasu.pseudo.paw.shape`.

Not read: ``exact_exchange_X_matrix`` (hybrid functionals), the kinetic-energy densities
(meta-GGA), ``lamb_shielding``, and ``kresse_joubert_local_ionic_pseudopotential`` -- the
Blochl formulation with ``zero_potential`` is the one amaterasu implements, and every
PAW-XML file carries that.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..base import RadialGrid
from ..paw.dataset import PartialWave, PAWDataset

__all__ = ["radial_grid_from_element", "read_pawxml"]

#: ``xc_functional`` name -> amaterasu functional name.
_XC_NAMES = {
    "PBE": "PBE",
    "PBESOL": "PBEsol",
    "PW": "LDA-PW",
    "LDA": "LDA-PW",
    "PW91": None,
    "REVPBE": None,
    "RPBE": None,
    "BLYP": None,
}


def _floats(element: ET.Element | None, name: str, source: str) -> np.ndarray:
    if element is None or element.text is None:
        raise ValueError(f"{source}: required element <{name}> is missing or empty")
    return np.fromstring(element.text, sep=" ")


def _attr(element: ET.Element, name: str) -> float:
    return float(element.attrib[name].strip())


def radial_grid_from_element(element: ET.Element, source: str = "") -> RadialGrid:
    """Build a :class:`RadialGrid` from a ``<radial_grid>`` element.

    All six equations of the specification are supported. The mesh is generated from the
    equation rather than from the optional ``<values>`` child so that ``dr/di`` is
    analytic and exactly consistent with it; when the file does tabulate the values, the
    two are compared and a disagreement is an error rather than a silent preference.
    """
    equation = element.attrib["eq"].strip()
    istart, iend = int(element.attrib["istart"]), int(element.attrib["iend"])
    i = np.arange(istart, iend + 1, dtype=float)

    if equation == "r=d*i":
        d = _attr(element, "d")
        r, dr_di = d * i, np.full_like(i, d)
    elif equation == "r=a*exp(d*i)":
        a, d = _attr(element, "a"), _attr(element, "d")
        r = a * np.exp(d * i)
        dr_di = d * r
    elif equation == "r=a*(exp(d*i)-1)":
        a, d = _attr(element, "a"), _attr(element, "d")
        r = a * (np.exp(d * i) - 1.0)
        dr_di = a * d * np.exp(d * i)
    elif equation == "r=a*i/(1-b*i)":
        a, b = _attr(element, "a"), _attr(element, "b")
        denominator = 1.0 - b * i
        r = a * i / denominator
        dr_di = a / denominator**2
    elif equation == "r=a*i/(n-i)":
        a, n = _attr(element, "a"), _attr(element, "n")
        denominator = n - i
        r = a * i / denominator
        dr_di = a * n / denominator**2
    elif equation == "r=(i/n+a)^5/a-a^4":
        a, n = _attr(element, "a"), _attr(element, "n")
        base = i / n + a
        r = base**5 / a - a**4
        dr_di = 5.0 * base**4 / (a * n)
    else:
        raise ValueError(f"{source}: unsupported radial_grid equation {equation!r}")

    tabulated = element.find("values")
    if tabulated is not None and tabulated.text:
        values = np.fromstring(tabulated.text, sep=" ")
        if len(values) == len(r):
            scale = max(float(np.max(np.abs(values))), 1e-30)
            deviation = float(np.max(np.abs(values - r))) / scale
            if deviation > 1e-9:
                raise ValueError(
                    f"{source}: the tabulated radial mesh disagrees with the equation "
                    f"{equation!r} by {deviation:.2e} relative; the grid parameters or the "
                    "index range are being misread"
                )
    return RadialGrid(r=np.ascontiguousarray(r), kind="log", dr_di=np.ascontiguousarray(dr_di))


def _on_grid(element: ET.Element, grids: dict[str, RadialGrid], main: str, source: str):
    """Read a radial function, padding to the main grid if it is stored truncated."""
    values = _floats(element, element.tag, source)
    grid_id = element.attrib.get("grid", main)
    if grid_id != main:
        raise NotImplementedError(
            f"{source}: <{element.tag}> lives on grid {grid_id!r} but the partial waves "
            f"use {main!r}; amaterasu does not resample between grids in one dataset"
        )
    size = len(grids[main])
    if len(values) > size:
        raise ValueError(f"{source}: <{element.tag}> has {len(values)} values, grid has {size}")
    if len(values) < size:
        # Some generators stop writing a localised function once it is identically zero.
        values = np.concatenate([values, np.zeros(size - len(values))])
    return np.ascontiguousarray(values)


def read_pawxml(path: str | os.PathLike) -> PAWDataset:
    """Read a PAW-XML v0.7 file into a :class:`PAWDataset`."""
    path = Path(path).expanduser()
    source = path.name
    text = (
        __import__("gzip").open(path, "rt", encoding="utf-8", errors="replace").read()
        if path.suffix == ".gz"
        else path.read_text(encoding="utf-8", errors="replace")
    )
    root = ET.fromstring(text)
    if root.tag != "paw_dataset":
        raise ValueError(f"{source}: root element is <{root.tag}>, expected <paw_dataset>")
    version = root.attrib.get("version", "?")

    atom = root.find("atom")
    if atom is None:
        raise ValueError(f"{source}: no <atom> element")
    symbol = atom.attrib["symbol"].strip()
    atomic_number = _attr(atom, "Z")
    core_electrons = _attr(atom, "core")
    valence_charge = _attr(atom, "valence")

    grids = {}
    for element in root.findall("radial_grid"):
        grids[element.attrib.get("id", "g1")] = radial_grid_from_element(element, source)
    if not grids:
        raise ValueError(f"{source}: no <radial_grid> element")

    waves_by_id: dict[str, dict] = {}
    states = root.find("valence_states")
    if states is None:
        raise ValueError(f"{source}: no <valence_states> element")
    order: list[str] = []
    for state in states:
        identifier = state.attrib["id"].strip()
        order.append(identifier)
        principal = state.attrib.get("n")
        waves_by_id[identifier] = {
            "l": int(state.attrib["l"]),
            "n": int(principal) if principal is not None and principal.strip() else None,
            "occupation": float(state.attrib.get("f", 0.0)),
            "energy": float(state.attrib.get("e", 0.0)),
            "label": identifier,
        }

    # The grid every partial wave lives on is the one the radial functions reference.
    tagged = root.find("ae_partial_wave")
    main = tagged.attrib.get("grid", next(iter(grids))) if tagged is not None else next(iter(grids))

    for tag, key in (
        ("ae_partial_wave", "ae"),
        ("pseudo_partial_wave", "ps"),
        ("projector_function", "projector"),
    ):
        for element in root.findall(tag):
            identifier = element.attrib["state"].strip()
            if identifier not in waves_by_id:
                raise ValueError(f"{source}: <{tag}> references unknown state {identifier!r}")
            waves_by_id[identifier][key] = _on_grid(element, grids, main, source)

    waves = []
    for identifier in order:
        entry = waves_by_id[identifier]
        missing = [k for k in ("ae", "ps", "projector") if k not in entry]
        if missing:
            raise ValueError(f"{source}: state {identifier!r} has no {', '.join(missing)}")
        waves.append(PartialWave(**entry))

    radius = root.find("paw_radius")
    if radius is not None:
        rcut = _attr(radius, "rc")
    else:  # pragma: no cover - every published dataset carries paw_radius
        rcut = max(float(s.attrib.get("rc", 0.0)) for s in states)

    kinetic = _floats(root.find("kinetic_energy_differences"), "kinetic_energy_differences", source)
    count = len(waves)
    if kinetic.size != count * count:
        raise ValueError(
            f"{source}: kinetic_energy_differences holds {kinetic.size} numbers, expected "
            f"{count}x{count}"
        )

    functional_element = root.find("xc_functional")
    raw_name = functional_element.attrib.get("name", "") if functional_element is not None else ""
    kind = functional_element.attrib.get("type", "") if functional_element is not None else ""
    functional = _XC_NAMES.get(raw_name.strip().upper())

    hints = {}
    ecut = root.find("pw_ecut")
    if ecut is not None:
        hints = {k: float(v) for k, v in ecut.attrib.items()}

    generator = root.find("generator")
    shape = root.find("shape_function")

    dataset = PAWDataset(
        symbol=symbol,
        atomic_number=atomic_number,
        core_electrons=core_electrons,
        valence_charge=valence_charge,
        grid=grids[main],
        waves=waves,
        rcut=rcut,
        kinetic_differences=kinetic.reshape(count, count),
        ae_core_density=_on_grid(root.find("ae_core_density"), grids, main, source),
        pseudo_core_density=_on_grid(root.find("pseudo_core_density"), grids, main, source),
        pseudo_valence_density=_on_grid(root.find("pseudo_valence_density"), grids, main, source),
        zero_potential=_on_grid(root.find("zero_potential"), grids, main, source),
        xc_functional=functional,
        xc_description=f"{kind} {raw_name}".strip() or "unspecified",
        generator=(generator.attrib.get("name", "") if generator is not None else ""),
        source=str(path),
        ecut_hints=hints,
        extra={
            "format": "PAW-XML",
            "version": version,
            "shape_function": dict(shape.attrib) if shape is not None else {},
            "generator_type": (generator.attrib.get("type", "") if generator is not None else ""),
        },
    )
    return dataset
