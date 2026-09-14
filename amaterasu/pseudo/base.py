"""Common representation of a norm-conserving pseudopotential.

Every reader produces one of these, so the rest of the code never sees a file format.
All quantities are in Hartree atomic units: radii in Bohr, potentials and projector
energies in Hartree.

Radial function conventions
---------------------------
These are the conventions the ABINIT psp8 and UPF formats actually use, verified
numerically against the PseudoDojo tables rather than assumed:

``local_potential``
    ``V_loc(r)`` in Hartree. Outside the core it must approach ``-Z_ion/r``;
    :meth:`NCPseudopotential.local_potential_tail_error` measures how well it does, which
    catches a misread valence charge or a units slip immediately.
``projectors``
    ``r beta_l(r)``, normalised so that ``integral p(r)^2 dr = 1`` (note: ``dr``, not
    ``r^2 dr`` -- the factor of ``r`` is already inside the stored function).
``valence_density`` and ``core_density``
    ``4 pi r^2 rho(r)``, so the enclosed charge is simply ``integral rho_stored(r) dr``.
    This is UPF's native convention; psp8 stores ``4 pi rho(r)`` instead and its reader
    multiplies by ``r^2`` on load. Storing the ``r^2`` factor avoids ever dividing by
    ``r``, which matters because some UPF meshes start at exactly ``r = 0``.

    The valence density integrates to ``Z_ion`` up to the truncation of the radial grid
    -- about 1% short for silicon's diffuse 3s3p on a 6 Bohr mesh, and exact on
    pslibrary's 100 Bohr mesh. The core density is a *pseudised* partial core for the
    nonlinear core correction and deliberately does **not** integrate to the number of
    core electrons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import simpson

__all__ = ["NCPseudopotential", "Projector", "RadialGrid", "decode_abinit_pspxc"]

#: ABINIT's built-in exchange-correlation identifiers (positive ``pspxc`` values).
_ABINIT_XC_CODES = {
    1: "LDA-Teter93",
    2: "LDA-PZ",
    3: "LDA-Teter91",
    4: "LDA-Wigner",
    5: "LDA-HedinLundqvist",
    6: "LDA-Xalpha",
    7: "LDA-PW",
    11: "PBE",
    12: "revPBE",
    14: "RPBE",
    15: "HCTH93",
    23: "WC",
}

#: libxc functional ids amaterasu recognises, for decoding negative ``pspxc`` values.
_LIBXC_IDS = {
    1: "LDA_X",
    9: "LDA_C_PZ",
    12: "LDA_C_PW",
    13: "LDA_C_PW_MOD",
    101: "GGA_X_PBE",
    102: "GGA_X_PBE_R",
    116: "GGA_X_PBE_SOL",
    117: "GGA_X_RPBE",
    130: "GGA_C_PBE",
    133: "GGA_C_PBE_SOL",
}

#: libxc (exchange, correlation) id pairs that correspond to an amaterasu functional name.
_LIBXC_PAIRS = {
    (1, 9): "LDA-PZ",
    (1, 12): "LDA-PW",
    (1, 13): "LDA-PW",
    (101, 130): "PBE",
    (116, 133): "PBEsol",
}


def decode_abinit_pspxc(pspxc: int) -> tuple[str | None, str]:
    """Turn an ABINIT ``pspxc`` code into ``(amaterasu functional name, description)``.

    A positive code is one of ABINIT's built-in functionals. A negative code packs two
    libxc identifiers as ``-(1000 * id_exchange + id_correlation)``: PBEsol, for instance,
    is ``-116133`` for ``GGA_X_PBE_SOL`` (116) with ``GGA_C_PBE_SOL`` (133).

    The name is ``None`` when the functional is one amaterasu does not implement; the
    description is always human-readable, so a mismatch can be reported usefully.

    >>> decode_abinit_pspxc(11)
    ('PBE', 'PBE')
    >>> decode_abinit_pspxc(-116133)
    ('PBEsol', 'libxc GGA_X_PBE_SOL + GGA_C_PBE_SOL')
    """
    if pspxc >= 0:
        name = _ABINIT_XC_CODES.get(pspxc)
        if name is None:
            return None, f"ABINIT pspxc {pspxc} (unrecognised)"
        return (name if name in ("LDA-PZ", "LDA-PW", "PBE") else None), name

    packed = -pspxc
    exchange_id, correlation_id = divmod(packed, 1000)
    exchange = _LIBXC_IDS.get(exchange_id, f"libxc:{exchange_id}")
    correlation = _LIBXC_IDS.get(correlation_id, f"libxc:{correlation_id}")
    return (
        _LIBXC_PAIRS.get((exchange_id, correlation_id)),
        f"libxc {exchange} + {correlation}",
    )


@dataclass(frozen=True)
class RadialGrid:
    """A radial mesh in Bohr, with the quadrature used for every radial integral."""

    r: np.ndarray
    #: ``linear`` for psp8's uniform mesh, ``log`` for the UPF exponential meshes.
    kind: str = "linear"
    #: For a UPF log mesh, ``dr/di``; ``None`` when the spacing is uniform.
    dr_di: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.r)

    @property
    def spacing(self) -> float:
        """Uniform step size; only meaningful for a linear mesh."""
        return float(self.r[1] - self.r[0])

    def integrate(self, values: np.ndarray, *, axis: int = -1):
        """``integral f(r) dr`` by Simpson's rule, over the last axis of ``values``.

        On a logarithmic mesh the integration variable is the *index*, with the tabulated
        ``dr/di`` as the Jacobian. Integrating in ``r`` instead would be only second-order
        accurate because the spacing varies by four orders of magnitude across the mesh;
        datasets ship ``dr/di`` precisely so that this can be done properly. The result is
        a plain float for one-dimensional input and an array otherwise.
        """
        values = np.asarray(values, dtype=float)
        if self.dr_di is not None and len(self.dr_di) == len(self.r):
            total = simpson(values * self.dr_di, dx=1.0, axis=axis)
        else:
            total = simpson(values, x=self.r, axis=axis)
        return float(total) if values.ndim == 1 else total

    def enclosed_charge(self, density: np.ndarray) -> float:
        """``integral 4 pi r^2 rho(r) dr``: the charge, for the stored convention."""
        return self.integrate(density)


@dataclass(frozen=True)
class Projector:
    """One Kleinman-Bylander projector: ``r beta_l(r)`` with its energy."""

    l: int
    index: int
    #: KB energy in Hartree; its sign says whether the channel is attractive or repulsive.
    energy: float
    #: ``r beta_l(r)`` on the radial grid.
    values: np.ndarray

    def norm(self, grid: RadialGrid) -> float:
        """``integral p(r)^2 dr``, which the generators normalise to 1."""
        return grid.integrate(self.values**2)


@dataclass
class NCPseudopotential:
    """A norm-conserving pseudopotential in the Kleinman-Bylander separable form."""

    symbol: str
    atomic_number: float
    valence_charge: float
    grid: RadialGrid
    local_potential: np.ndarray
    projectors: list[Projector]
    l_max: int
    l_local: int
    #: amaterasu functional name the dataset was generated with, or None if unrecognised.
    xc_functional: str | None = None
    #: Human-readable description of the generating functional, always populated.
    xc_description: str = ""
    #: Pseudised partial core ``4 pi r^2 rho_c(r)`` for the nonlinear core correction.
    core_density: np.ndarray | None = None
    #: Atomic valence pseudo-density ``4 pi r^2 rho_v(r)``; the starting guess for the SCF.
    valence_density: np.ndarray | None = None
    #: ``"NC"`` or ``"PAW"``; only norm-conserving projectors are unit-normalised.
    kind: str = "NC"
    #: Full ``D_ij`` coupling matrix in Hartree, shape ``(nproj, nproj)``. Diagonal for a
    #: norm-conserving dataset, where it reduces to the Kleinman-Bylander energies; a PAW
    #: dataset couples projectors of the same ``l`` off-diagonally.
    dij: np.ndarray | None = None
    generator: str = ""
    source: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("local_potential", "core_density", "valence_density"):
            value = getattr(self, name)
            if value is not None and len(value) != len(self.grid):
                raise ValueError(
                    f"{name} has {len(value)} points but the grid has {len(self.grid)}"
                )

    def __str__(self) -> str:
        channels = ", ".join(f"l={l}:{self.n_projectors(l)}" for l in range(self.l_max + 1))
        return (
            f"{self.symbol} (Z={self.atomic_number:g}, Z_val={self.valence_charge:g}) "
            f"{self.xc_description or 'unknown XC'}, {len(self.projectors)} projectors "
            f"[{channels}], {len(self.grid)} grid points to r={self.grid.r[-1]:.2f} Bohr"
        )

    @property
    def has_core_correction(self) -> bool:
        """True if the dataset carries a partial core density for the NLCC."""
        return self.core_density is not None

    def n_projectors(self, l: int) -> int:
        return sum(1 for p in self.projectors if p.l == l)

    def projectors_for(self, l: int) -> list[Projector]:
        return [p for p in self.projectors if p.l == l]

    # ----------------------------------------------------------------- diagnostics
    def local_potential_tail_error(self, r_min: float = 3.0) -> float:
        """Largest relative deviation of ``V_loc`` from ``-Z_ion/r`` beyond ``r_min``.

        Outside the core region the pseudopotential is the bare ionic Coulomb tail, so
        this must be small. It is the cheapest check that the valence charge, the units
        and the grid were all read correctly: a wrong ``Z_ion`` shows up immediately.
        """
        outside = self.grid.r >= r_min
        if not np.any(outside):
            raise ValueError(f"the grid does not extend to r_min = {r_min} Bohr")
        expected = -self.valence_charge / self.grid.r[outside]
        return float(np.max(np.abs(self.local_potential[outside] - expected) / np.abs(expected)))

    def integrated_valence_charge(self) -> float:
        """``integral 4 pi r^2 rho_v(r) dr``.

        This falls slightly short of ``Z_ion`` because the radial grid is truncated at a
        finite radius; the shortfall is largest for diffuse valence shells (about 1% for
        silicon's 3s3p, 0.02% for oxygen's compact 2p).
        """
        if self.valence_density is None:
            raise ValueError(f"{self.symbol}: this dataset carries no valence density")
        return self.grid.enclosed_charge(self.valence_density)

    def check(self, *, tail_tolerance: float = 1e-3, charge_tolerance: float = 0.05) -> list[str]:
        """Return a list of problems found; empty means the dataset looks sound."""
        problems = []
        tail = self.local_potential_tail_error()
        if tail > tail_tolerance:
            problems.append(
                f"V_loc deviates from -Z_ion/r by {tail:.2e} beyond r = 3 Bohr "
                f"(Z_ion = {self.valence_charge:g}); the valence charge or units may be wrong"
            )
        if self.valence_density is not None:
            charge = self.integrated_valence_charge()
            relative = abs(charge - self.valence_charge) / self.valence_charge
            if relative > charge_tolerance:
                problems.append(
                    f"the valence density integrates to {charge:.4f} but Z_ion = "
                    f"{self.valence_charge:g} ({relative:.1%} off)"
                )
        if self.kind == "NC":
            # Norm-conserving generators normalise r*beta(r) to unity. PAW projectors are
            # dual to the partial waves and carry their scale in D_ij instead, so the same
            # check would be meaningless there.
            for projector in self.projectors:
                norm = projector.norm(self.grid)
                if abs(norm - 1.0) > 1e-3:
                    problems.append(
                        f"projector l={projector.l} #{projector.index} has norm "
                        f"{norm:.6f}, not 1; the radial convention may be misread"
                    )
        return problems
