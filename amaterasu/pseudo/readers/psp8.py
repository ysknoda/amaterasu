"""Reader for the ABINIT ``psp8`` norm-conserving pseudopotential format.

``psp8`` is what the PseudoDojo ONCVPSP tables ship in, and it is the format amaterasu
targets first: the tables cover LDA, PBE and PBEsol, every dataset comes with a published
delta-gauge validation, and the files are CC BY 4.0.

File layout
-----------
Six header lines, then one block per angular momentum, then the local potential, then two
optional density blocks::

    title                                     element, generator, core radii
    zatom  zion  pspd                         nuclear and valence charge, generation date
    pspcod  pspxc  lmax  lloc  mmax  r2well    format 8, XC code, l_max, local l, points
    rchrg  fchrg  qchrg                       model core charge parameters
    nproj[0..4]                               projectors per angular momentum
    extension_switch                          1 or 3 adds a valence pseudo-density block

    for each l = 0..lmax, skipping l == lloc:
        l  ekb[0] ... ekb[nproj[l]-1]          KB energies, Hartree
        mmax rows:  index  r  p_0(r) ... p_n(r)

    lloc                                       header of the local potential block
    mmax rows:  index  r  V_loc(r)

    if fchrg > 0:  mmax rows: index r  rho_core and its derivatives
    if extension_switch in (1, 3): mmax rows: index r  rho_valence and its derivatives

Numbers use Fortran ``D`` exponents. Trailing lines after the data are the generator's
own log and are ignored.

Conventions (all verified numerically against the PseudoDojo tables, see the tests):
``V_loc`` approaches ``-Z_ion/r`` and is already in Hartree (unlike UPF, which uses
Rydberg); projectors are ``r beta(r)`` normalised to ``integral p^2 dr = 1``; densities are
stored in the file as ``4 pi rho(r)`` and are multiplied by ``r^2`` on load to match the
internal convention.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ..base import NCPseudopotential, Projector, RadialGrid, decode_abinit_pspxc

__all__ = ["read_psp8"]

_SPIN_ORBIT_SWITCHES = (2, 3)
_VALENCE_DENSITY_SWITCHES = (1, 3)


def _read_text(path: Path) -> str:
    """Read a file, transparently decompressing ``.gz``."""
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _to_float(token: str) -> float:
    """Parse a Fortran real, which may use ``D`` rather than ``E`` for the exponent."""
    return float(token.replace("D", "E").replace("d", "e"))


def _read_table(lines: list[str], start: int, rows: int, source: str) -> np.ndarray:
    """Read ``rows`` numbered data lines, dropping the leading index column."""
    table = []
    for offset in range(rows):
        index = start + offset
        if index >= len(lines):
            raise ValueError(
                f"{source}: file ends after {offset} of {rows} expected data rows "
                f"(is mmax correct?)"
            )
        tokens = lines[index].split()
        if len(tokens) < 2:
            raise ValueError(f"{source}: line {index + 1} is not a data row: {lines[index]!r}")
        table.append([_to_float(t) for t in tokens[1:]])
    widths = {len(row) for row in table}
    if len(widths) != 1:
        raise ValueError(f"{source}: data block starting at line {start + 1} has ragged columns")
    return np.array(table, dtype=float)


def read_psp8(path: str | os.PathLike) -> NCPseudopotential:
    """Read an ABINIT ``psp8`` file into an :class:`NCPseudopotential`."""
    path = Path(path)
    source = path.name
    lines = _read_text(path).splitlines()
    if len(lines) < 7:
        raise ValueError(f"{source}: too short to be a psp8 file ({len(lines)} lines)")

    title = lines[0].strip()
    symbol = title.split()[0] if title else path.stem
    generator = title

    atomic_number = _to_float(lines[1].split()[0])
    valence_charge = _to_float(lines[1].split()[1])

    header = lines[2].split()
    pspcod, pspxc = int(header[0]), int(header[1])
    l_max, l_local, mmax = int(header[2]), int(header[3]), int(header[4])
    if pspcod != 8:
        raise ValueError(f"{source}: pspcod is {pspcod}, not 8; this is not a psp8 file")
    if mmax < 2:
        raise ValueError(f"{source}: mmax = {mmax} is too small for a radial mesh")

    fchrg = _to_float(lines[3].split()[1])
    n_projectors = [int(x) for x in lines[4].split()[:5]]
    extension_switch = int(lines[5].split()[0])
    if extension_switch in _SPIN_ORBIT_SWITCHES:
        raise NotImplementedError(
            f"{source}: extension_switch = {extension_switch} means the file carries "
            "spin-orbit projectors, which amaterasu does not read yet. Use the "
            "scalar-relativistic (SR) table instead of the fully-relativistic (FR) one."
        )

    cursor = 6
    projectors: list[Projector] = []
    reference_r: np.ndarray | None = None

    for l in range(l_max + 1):
        if l == l_local:
            continue  # the local channel has no projectors; it is written out separately
        count = n_projectors[l] if l < len(n_projectors) else 0
        if count == 0:
            continue
        tokens = lines[cursor].split()
        if int(tokens[0]) != l:
            raise ValueError(
                f"{source}: expected the l = {l} projector block at line {cursor + 1}, "
                f"found {lines[cursor]!r}"
            )
        energies = [_to_float(t) for t in tokens[1 : 1 + count]]
        if len(energies) != count:
            raise ValueError(
                f"{source}: l = {l} declares {count} projectors but its header carries "
                f"{len(energies)} KB energies"
            )
        table = _read_table(lines, cursor + 1, mmax, source)
        radii, values = table[:, 0], table[:, 1:]
        if values.shape[1] < count:
            raise ValueError(
                f"{source}: l = {l} has {values.shape[1]} projector columns, expected {count}"
            )
        reference_r = radii if reference_r is None else reference_r
        if not np.allclose(radii, reference_r, atol=1e-12):
            raise ValueError(f"{source}: the l = {l} block uses a different radial mesh")
        projectors.extend(
            Projector(l=l, index=j, energy=energies[j], values=np.ascontiguousarray(values[:, j]))
            for j in range(count)
        )
        cursor += 1 + mmax

    # --- local potential -------------------------------------------------------
    tokens = lines[cursor].split()
    if int(tokens[0]) != l_local:
        raise ValueError(
            f"{source}: expected the local (l = {l_local}) block at line {cursor + 1}, "
            f"found {lines[cursor]!r}"
        )
    table = _read_table(lines, cursor + 1, mmax, source)
    radii, local_potential = table[:, 0], table[:, 1]
    cursor += 1 + mmax
    if reference_r is not None and not np.allclose(radii, reference_r, atol=1e-12):
        raise ValueError(f"{source}: the local potential uses a different radial mesh")

    grid = RadialGrid(r=np.ascontiguousarray(radii), kind="linear")

    # --- optional density blocks, which carry no header line --------------------
    # psp8 stores 4 pi rho(r); amaterasu keeps 4 pi r^2 rho(r) internally (see base.py),
    # so the r^2 goes on here rather than at every use site.
    r_squared = grid.r**2
    core_density = None
    if fchrg > 0.0:
        core_density = np.ascontiguousarray(
            _read_table(lines, cursor, mmax, source)[:, 1] * r_squared
        )
        cursor += mmax

    valence_density = None
    if extension_switch in _VALENCE_DENSITY_SWITCHES:
        valence_density = np.ascontiguousarray(
            _read_table(lines, cursor, mmax, source)[:, 1] * r_squared
        )
        cursor += mmax

    functional, description = decode_abinit_pspxc(pspxc)
    core_radii = _parse_core_radii(title)

    return NCPseudopotential(
        symbol=symbol,
        atomic_number=atomic_number,
        valence_charge=valence_charge,
        grid=grid,
        local_potential=np.ascontiguousarray(local_potential),
        projectors=projectors,
        l_max=l_max,
        l_local=l_local,
        kind="NC",
        xc_functional=functional,
        xc_description=description,
        core_density=core_density,
        valence_density=valence_density,
        generator=generator,
        source=str(path),
        extra={
            "format": "psp8",
            "pspxc": pspxc,
            "fchrg": fchrg,
            "extension_switch": extension_switch,
            "n_projectors": n_projectors,
            "core_radii": core_radii,
        },
    )


def _parse_core_radii(title: str) -> list[float]:
    """Pull the ``r_core=`` values out of an ONCVPSP title line, if present."""
    match = re.search(r"r_core\s*=\s*(.*)$", title)
    if not match:
        return []
    radii = []
    for token in match.group(1).split():
        try:
            radii.append(float(token))
        except ValueError:
            break
    return radii
