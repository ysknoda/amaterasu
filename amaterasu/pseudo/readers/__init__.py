"""Pseudopotential file readers, with format detection.

Every reader returns an :class:`~amaterasu.pseudo.base.NCPseudopotential`, so the rest of
amaterasu never branches on file format. ``.gz`` files are decompressed transparently.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

from ..base import NCPseudopotential
from ..paw.dataset import PAWDataset
from .pawxml import read_pawxml
from .psp8 import read_psp8
from .upf import read_upf
from .upfpaw import read_upf_paw

__all__ = [
    "detect_format",
    "read_pawxml",
    "read_pseudopotential",
    "read_psp8",
    "read_upf",
    "read_upf_paw",
]


def _read_upf_either(path: Path) -> NCPseudopotential | PAWDataset:
    """A UPF file can hold either kind; the header says which.

    ``pseudo_type="PAW"`` means a Kresse-Joubert dataset with partial waves and
    augmentation, which :func:`~amaterasu.pseudo.readers.upfpaw.read_upf_paw` converts into
    the Blochl form the rest of the code works in. Anything else is norm-conserving (or
    ultrasoft, which is read for its norm-conserving content only).
    """
    import xml.etree.ElementTree as ET

    from .upf import _parse_xml

    root: ET.Element = _parse_xml(path)
    header = root.find("PP_HEADER")
    kind = (header.attrib.get("pseudo_type", "") if header is not None else "").strip().upper()
    return read_upf_paw(path) if kind == "PAW" else read_upf(path)


_READERS = {"psp8": read_psp8, "upf": _read_upf_either, "pawxml": read_pawxml}


def _peek(path: Path, size: int = 4096) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read(size)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(size)


def detect_format(path: str | os.PathLike) -> str:
    """Identify the pseudopotential format of ``path`` from its content.

    The extension alone is unreliable -- ``.upf``, ``.UPF``, ``.psp8`` and bare names all
    occur in the wild -- so the file's own first lines decide. UPF announces itself with a
    ``<UPF`` root element; psp8 declares ``pspcod = 8`` on its third line.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such pseudopotential file: {path}")
    head = _peek(path)
    if "<paw_dataset" in head:
        return "pawxml"
    if "<UPF" in head or "<PP_HEADER" in head:
        return "upf"
    lines = head.splitlines()
    if len(lines) >= 3:
        tokens = lines[2].split()
        if tokens and tokens[0] == "8":
            return "psp8"
    raise ValueError(
        f"{path.name}: unrecognised pseudopotential format. Expected PAW-XML (a "
        "<paw_dataset> root), UPF (a <UPF> root) or ABINIT psp8 (pspcod = 8 on the "
        "third line)."
    )


def read_pseudopotential(
    path: str | os.PathLike, *, format: str | None = None
) -> NCPseudopotential | PAWDataset:
    """Read a pseudopotential, detecting the format unless ``format`` is given."""
    path = Path(path).expanduser()
    resolved = format or detect_format(path)
    reader = _READERS.get(resolved)
    if reader is None:
        raise ValueError(f"unknown pseudopotential format {resolved!r}; expected psp8 or upf")
    return reader(path)
