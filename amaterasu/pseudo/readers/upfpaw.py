"""Reading a pslibrary ``*-kjpaw`` UPF file as a PAW dataset.

This is the only route to PBEsol + PAW: the PAW-XML tables (JTH, GPAW) cover PBE and LDA
only, while pslibrary publishes ``pz``, ``pbe`` and ``pbesol`` in UPF form. The file is a
Kresse-Joubert dataset, so two of the quantities the Blochl-form
:class:`~amaterasu.pseudo.paw.dataset.PAWDataset` needs are not in it and have to be
derived. Both derivations are checked in the test suite against a file that *does* ship the
answer.

``v_bar``
    Not stored. Derived from the local ionic pseudopotential by the transformation in the
    PAW-XML specification; see :mod:`amaterasu.pseudo.paw.kresse_joubert`.
``kinetic_energy_differences``
    Not stored either. Recomputed from the partial waves by integrating the radial kinetic
    operator by parts. The surface term that leaves behind vanishes identically, because a
    PAW partial wave and its smooth counterpart are *bitwise* equal outside the matching
    radius in the files checked here.

Conventions, established by measurement against the file rather than assumed
---------------------------------------------------------------------------
``PP_R``, ``PP_RAB``
    Bohr, and the tabulated ``dr/di`` of the logarithmic mesh.
``PP_AEWFC.i``, ``PP_PSWFC.i``, ``PP_BETA.i``
    ``r`` times the radial function. Verified: the bound all-electron waves integrate to
    ``int |r phi|^2 dr = 1``, and ``int (r beta_i)(r phi~_j) dr`` is the identity within
    each ``l`` channel to 1e-4 (2.6e-2 for silicon's auxiliary ``d`` channel).
``PP_LOCAL``, ``PP_AE_VLOC``
    Rydberg. Both approach ``-2 Z_val / r``: ``PP_AE_VLOC`` is the potential of the nucleus
    *plus the all-electron core*, not of the bare nucleus.
``PP_NLCC``, ``PP_AE_NLCC``
    Bare densities. Verified by integration: ``4 pi int r^2 PP_AE_NLCC dr`` is exactly the
    core electron count, and ``PP_NLCC`` gives the smaller smooth core charge.
``PP_RHOATOM``
    Already ``4 pi r^2 rho``; it integrates to ``z_valence`` exactly.
``PP_AUGMENTATION``
    ``cutoff_r_index`` marks the augmentation radius. It sits outside the radius where the
    all-electron and smooth partial waves start to agree, which is the property the
    one-centre terms rely on.

The augmentation functions (``PP_QIJL``) are *not* read: amaterasu builds its own
compensation charge from the multipole moments of the partial waves, which is the Blochl
prescription and does not need the dataset's own ``Q_ij``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ...units import RYDBERG_IN_HARTREE
from ...xc.driver import XCFunctional
from ..base import RadialGrid
from ..paw.dataset import PartialWave, PAWDataset
from ..paw.kresse_joubert import ROOT_FOUR_PI, zero_potential_from_kresse_joubert
from ..paw.radial import derivative, integrate
from ..paw.shape import GaussianShape
from .upf import _atomic_number, _decode_functional, _parse_xml, _text_array

__all__ = ["kinetic_energy_differences", "read_upf_paw"]

_SHELL = re.compile(r"^\s*(\d+)([SPDFGH])", re.IGNORECASE)


def _indexed(parent, prefix: str) -> list:
    """Children named ``prefix.1``, ``prefix.2``, ... in index order."""
    found = [e for e in parent if e.tag.startswith(f"{prefix}.")]
    return sorted(found, key=lambda e: int(e.tag.split(".")[1]))


def kinetic_energy_differences(
    grid: RadialGrid, ae: np.ndarray, ps: np.ndarray, angular: np.ndarray, stop: int
) -> np.ndarray:
    """``<phi_i|T|phi_j> - <phi~_i|T|phi~_j>`` in Hartree, from the partial waves.

    With ``u = r phi`` the radial kinetic energy is

        <phi_i|T|phi_j> = (1/2) int [ u_i' u_j' + l(l+1) u_i u_j / r^2 ] dr ,

    the integrated-by-parts form, which needs only a first derivative of a function that is
    tabulated rather than a second derivative of one that has a cusp at the nucleus. The
    surface term at ``r_c`` cancels between the two halves because the partial waves match
    there. ``T`` is diagonal in the angular part, so only pairs sharing an ``l`` contribute.
    """
    r = grid.r[:stop]
    inverse = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
    count = len(angular)
    out = np.zeros((count, count))
    for radial, sign in ((ae, 1.0), (ps, -1.0)):
        u = radial[:, :stop] * r
        du = derivative(grid, u)
        for i in range(count):
            for j in range(count):
                if angular[i] != angular[j]:
                    continue
                l = int(angular[i])
                integrand = du[i] * du[j] + l * (l + 1) * u[i] * u[j] * inverse**2
                out[i, j] += sign * 0.5 * integrate(grid, integrand, upto=stop)
    return out


def read_upf_paw(path: str | os.PathLike) -> PAWDataset:
    """Read a Kresse-Joubert UPF-PAW file into a Blochl-form :class:`PAWDataset`."""
    path = Path(path)
    source = path.name
    root = _parse_xml(path)
    header = root.find("PP_HEADER")
    if header is None:
        raise ValueError(f"{source}: no PP_HEADER block")
    attributes = {k: (v.strip() if isinstance(v, str) else v) for k, v in header.attrib.items()}
    if attributes.get("pseudo_type", "").upper() != "PAW":
        raise ValueError(f"{source}: pseudo_type is {attributes.get('pseudo_type')!r}, not PAW")
    if attributes.get("has_so", "F").upper() in ("T", "TRUE", ".TRUE."):
        raise NotImplementedError(
            f"{source}: this dataset carries spin-orbit projectors, which amaterasu does "
            "not read yet; use the scalar-relativistic variant"
        )

    symbol = attributes.get("element", path.stem.split(".")[0]).strip()
    atomic_number = _atomic_number(symbol, attributes)
    valence_charge = float(attributes["z_valence"])

    mesh = root.find("PP_MESH")
    radii = _text_array(mesh.find("PP_R") if mesh is not None else None, "PP_MESH/PP_R", source)
    dr_di = _text_array(mesh.find("PP_RAB"), "PP_MESH/PP_RAB", source)
    grid = RadialGrid(r=np.ascontiguousarray(radii), kind="log", dr_di=dr_di)

    augmentation = root.find(".//PP_AUGMENTATION")
    if augmentation is None:
        raise ValueError(
            f"{source}: the file says pseudo_type=PAW but carries no PP_AUGMENTATION block"
        )
    sphere_index = int(augmentation.attrib["cutoff_r_index"])
    rcut = float(radii[sphere_index])
    stop = sphere_index + 1

    nonlocal_block = root.find("PP_NONLOCAL")
    full = root.find("PP_FULL_WFC")
    if nonlocal_block is None or full is None:
        raise ValueError(f"{source}: a PAW dataset needs both PP_NONLOCAL and PP_FULL_WFC")
    betas = _indexed(nonlocal_block, "PP_BETA")
    ae_blocks = _indexed(full, "PP_AEWFC")
    ps_blocks = _indexed(full, "PP_PSWFC")
    if not (len(betas) == len(ae_blocks) == len(ps_blocks)):
        raise ValueError(
            f"{source}: {len(betas)} projectors but {len(ae_blocks)} all-electron and "
            f"{len(ps_blocks)} smooth partial waves; they must come in threes"
        )

    inverse_r = np.where(radii > 0, 1.0 / np.where(radii > 0, radii, 1.0), 0.0)

    def divided(element, name) -> np.ndarray:
        """``r f(r)`` as tabulated, turned into ``f(r)``, with the origin extrapolated."""
        values = _text_array(element, name, source) * inverse_r
        if radii[0] <= 0.0:
            values[0] = 2.0 * values[1] - values[2]
        return np.ascontiguousarray(values)

    occupations = root.find(".//PP_OCCUPATIONS")
    fillings = (
        _text_array(occupations, "PP_OCCUPATIONS", source) if occupations is not None else None
    )
    # PP_CHI carries a reference energy, but only for the bound atomic states; the
    # scattering channels have none, and nothing in the code needs one.
    energies = {}
    for chi in _indexed(root.find("PP_PSWFC") or root, "PP_CHI"):
        label = chi.attrib.get("label", "").strip().upper()
        if "pseudo_energy" in chi.attrib:
            energies[label] = float(chi.attrib["pseudo_energy"]) * RYDBERG_IN_HARTREE

    waves: list[PartialWave] = []
    triples = zip(betas, ae_blocks, ps_blocks, strict=True)
    for index, (beta, ae_block, ps_block) in enumerate(triples):
        label = (beta.attrib.get("label") or ae_block.attrib.get("label") or "").strip()
        l = int(beta.attrib["angular_momentum"])
        occupation = float(fillings[index]) if fillings is not None else 0.0
        match = _SHELL.match(label)
        principal = int(match.group(1)) if match and occupation > 0 else None
        waves.append(
            PartialWave(
                l=l,
                label=label or f"l={l}",
                n=principal,
                occupation=occupation,
                energy=energies.get(label.upper(), 0.0),
                ae=divided(ae_block, f"PP_AEWFC.{index + 1}"),
                ps=divided(ps_block, f"PP_PSWFC.{index + 1}"),
                projector=divided(beta, f"PP_BETA.{index + 1}"),
            )
        )

    angular = np.array([w.l for w in waves], dtype=int)
    ae = np.array([w.ae for w in waves])
    ps = np.array([w.ps for w in waves])
    kinetic = kinetic_energy_differences(grid, ae, ps, angular, stop)

    # Densities, converted to the Y_00 coefficients the Blochl code keeps everything in.
    pseudo_core = _text_array(root.find("PP_NLCC"), "PP_NLCC", source) * ROOT_FOUR_PI
    ae_core = _text_array(root.find(".//PP_AE_NLCC"), "PP_AE_NLCC", source) * ROOT_FOUR_PI
    rhoatom = _text_array(root.find("PP_RHOATOM"), "PP_RHOATOM", source)
    pseudo_valence = rhoatom * inverse_r**2 / ROOT_FOUR_PI
    if radii[0] <= 0.0:
        pseudo_valence[0] = 2.0 * pseudo_valence[1] - pseudo_valence[2]

    local = _text_array(root.find("PP_LOCAL"), "PP_LOCAL", source) * RYDBERG_IN_HARTREE
    kj_local = np.ascontiguousarray(local * ROOT_FOUR_PI)

    functional, description = _decode_functional(attributes.get("functional", ""))
    core_electrons = atomic_number - valence_charge
    zero_potential = zero_potential_from_kresse_joubert(
        XCFunctional(functional or "PBE"),
        grid,
        local_potential=kj_local,
        pseudo_core_density=pseudo_core,
        pseudo_valence_density=pseudo_valence,
        shape=GaussianShape.for_radius(rcut),
        core_electrons=core_electrons,
        valence_electrons=valence_charge,
        atomic_number=atomic_number,
    )
    # v_bar is short-ranged by construction; everything past the sphere is the residue of
    # cancelling two long-ranged potentials against each other.
    zero_potential = np.where(radii <= rcut, zero_potential, 0.0)

    hints = {}
    for key, attribute in (("low", "wfc_cutoff"), ("high", "rho_cutoff")):
        if attributes.get(attribute):
            hints[key] = float(attributes[attribute]) * RYDBERG_IN_HARTREE
    if "low" in hints:
        hints.setdefault("medium", hints["low"])

    return PAWDataset(
        symbol=symbol,
        atomic_number=atomic_number,
        core_electrons=core_electrons,
        valence_charge=valence_charge,
        grid=grid,
        waves=waves,
        rcut=rcut,
        kinetic_differences=kinetic,
        ae_core_density=np.ascontiguousarray(ae_core),
        pseudo_core_density=np.ascontiguousarray(pseudo_core),
        pseudo_valence_density=np.ascontiguousarray(pseudo_valence),
        zero_potential=np.ascontiguousarray(zero_potential),
        xc_functional=functional,
        xc_description=description,
        generator=attributes.get("generated", ""),
        source=str(path),
        ecut_hints=hints,
        extra={
            "format": "upf-paw",
            "formulation": "kresse-joubert",
            "upf_version": root.attrib.get("version", ""),
            "kj_local_potential": kj_local,
            "augmentation_shape": augmentation.attrib.get("shape", ""),
            "zero_potential_shape": str(GaussianShape.for_radius(rcut)),
        },
    )
