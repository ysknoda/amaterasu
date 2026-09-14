"""Brillouin-zone occupations: smearing functions, the Fermi level, and the entropy term.

A metal's Fermi surface cuts through bands, so a finite k-point mesh samples a
discontinuous occupation and the total energy converges agonisingly slowly with the mesh.
Smearing replaces the step with a smooth function of width ``sigma``, which converges far
faster -- at the price of introducing a fictitious electronic temperature.

Conventions
-----------
Occupations are per spin state and lie in ``[0, 1]``; a state carries ``g = 2/nspin``
electrons. The reduced energy is ``x = (eps - mu)/sigma``, so ``f -> 1`` well below the
Fermi level. Each scheme also defines a generalised entropy ``S(x) >= 0`` (except
Methfessel-Paxton, where it may take either sign) entering the free energy as

    F = E - sigma * sum_{nk} w_nk g S(x_nk).

Three energies are therefore reportable and they are *not* interchangeable:

``E``   the Kohn-Sham energy at the smeared occupations -- not variational;
``F``   the free energy, which *is* the quantity whose gradient gives the forces;
``E0``  the ``sigma -> 0`` extrapolation ``(E + F)/2``, valid for the symmetric schemes
        (Gaussian, Methfessel-Paxton, Marzari-Vanderbilt) but not for Fermi-Dirac, where
        the smearing is a physical temperature and ``F`` is the meaningful free energy.

The Methfessel-Paxton and cold-smearing recursions follow Methfessel and Paxton, PRB 40,
3616 (1989) and Marzari, Vanderbilt, De Vita and Payne, PRL 82, 3296 (1999); the
implementation mirrors the well-tested Quantum ESPRESSO ``wgauss``/``w1gauss`` recursions,
with the sign of ``x`` flipped to the convention above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

__all__ = ["OccupationResult", "compute_occupations", "entropy_function", "occupation_function"]

#: Largest argument passed to exp(); beyond it the occupations are 0 or 1 to machine
#: precision and the exponential would simply overflow.
_MAX_ARGUMENT = 200.0

SCHEMES = (
    "fermi-dirac",
    "gaussian",
    "methfessel-paxton",
    "marzari-vanderbilt",
    "tetrahedron",
    "fixed",
)


def _hermite_terms(x: np.ndarray, order: int):
    """Yield ``(i, a_i, H_{2i-1}(x) e^{-x^2}, H_{2i-2}(x) e^{-x^2}, 2i)`` for the MP sums."""
    arg = np.minimum(_MAX_ARGUMENT, x**2)
    hp = np.exp(-arg)
    hd = np.zeros_like(x)
    ni = 0
    a = 1.0 / np.sqrt(np.pi)
    for i in range(1, order + 1):
        hd = 2.0 * x * hp - 2.0 * ni * hd
        ni += 1
        a = -a / (i * 4.0)
        previous = hp
        hp = 2.0 * x * hd - 2.0 * ni * hp
        ni += 1
        yield a, hd, hp, previous, ni


def occupation_function(x: np.ndarray, scheme: str, order: int = 1) -> np.ndarray:
    """Occupation of a state at reduced energy ``x = (eps - mu)/sigma``."""
    x = np.asarray(x, dtype=float)
    if scheme == "fermi-dirac":
        return 1.0 / (1.0 + np.exp(np.clip(x, -_MAX_ARGUMENT, _MAX_ARGUMENT)))
    if scheme == "marzari-vanderbilt":
        shifted = x + 1.0 / np.sqrt(2.0)
        arg = np.minimum(_MAX_ARGUMENT, shifted**2)
        return 0.5 * erfc(shifted) + np.exp(-arg) / np.sqrt(2.0 * np.pi)
    if scheme in ("gaussian", "tetrahedron", "fixed"):
        return 0.5 * erfc(x)
    if scheme == "methfessel-paxton":
        # The recursion is written for the opposite sign convention.
        negated = -x
        result = 0.5 * erfc(x)
        for a, hd, _, _, _ in _hermite_terms(negated, order):
            result = result - a * hd
        return result
    raise ValueError(f"unknown smearing scheme {scheme!r}; available: {', '.join(SCHEMES)}")


def entropy_function(x: np.ndarray, scheme: str, order: int = 1) -> np.ndarray:
    """Generalised entropy ``S(x)``, with ``F = E - sigma sum w g S``."""
    x = np.asarray(x, dtype=float)
    if scheme == "fermi-dirac":
        f = occupation_function(x, scheme)
        safe = np.clip(f, 1e-300, 1.0 - 1e-16)
        return -(safe * np.log(safe) + (1.0 - safe) * np.log1p(-safe))
    if scheme == "marzari-vanderbilt":
        shifted = x + 1.0 / np.sqrt(2.0)
        arg = np.minimum(_MAX_ARGUMENT, shifted**2)
        return shifted * np.exp(-arg) / np.sqrt(2.0 * np.pi)
    if scheme in ("gaussian", "tetrahedron", "fixed"):
        return 0.5 * np.exp(-np.minimum(_MAX_ARGUMENT, x**2)) / np.sqrt(np.pi)
    if scheme == "methfessel-paxton":
        negated = -x
        result = 0.5 * np.exp(-np.minimum(_MAX_ARGUMENT, negated**2)) / np.sqrt(np.pi)
        for a, _, hp, previous, ni in _hermite_terms(negated, order):
            result = result + a * (0.5 * hp + ni * previous)
        return result
    raise ValueError(f"unknown smearing scheme {scheme!r}; available: {', '.join(SCHEMES)}")


@dataclass
class OccupationResult:
    """Occupations and the derived Fermi level and entropy."""

    #: Shape ``(nspin, nkpoints, nbands)``, each in ``[0, 1]``.
    occupations: np.ndarray
    #: Fermi level in Hartree.
    fermi_level: float
    #: ``sigma * sum w g S`` in Hartree; subtract it from ``E`` to get the free energy.
    entropy_term: float
    #: Highest occupied to lowest unoccupied separation in Hartree, or ``None`` for a metal.
    gap: float | None
    #: Total magnetic moment in Bohr magnetons (zero for a non-spin-polarised run).
    magnetization: float

    @property
    def is_metallic(self) -> bool:
        return self.gap is None


def _fill_fixed(
    eigenvalues: np.ndarray, weights: np.ndarray, nelectrons: float, degeneracy: float
) -> tuple[np.ndarray, float, float | None]:
    """Zero-temperature filling: occupy states in energy order until the electrons run out."""
    flat = eigenvalues.ravel()
    order = np.argsort(flat, kind="stable")
    state_weights = np.broadcast_to(weights[None, :, None] * degeneracy, eigenvalues.shape).ravel()[
        order
    ]
    cumulative = np.cumsum(state_weights)
    occupations = np.zeros_like(flat)
    full = cumulative <= nelectrons + 1e-12
    occupations[order[full]] = 1.0
    boundary = int(np.count_nonzero(full))
    if boundary < len(order):
        remaining = nelectrons - (cumulative[boundary - 1] if boundary else 0.0)
        occupations[order[boundary]] = max(0.0, remaining / state_weights[boundary])
    homo = flat[order[max(boundary - 1, 0)]]
    lumo = flat[order[boundary]] if boundary < len(order) else homo
    gap = lumo - homo
    return occupations.reshape(eigenvalues.shape), 0.5 * (homo + lumo), gap


def compute_occupations(
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    nelectrons: float,
    *,
    scheme: str = "gaussian",
    smearing: float = 0.0,
    order: int = 1,
    nspin: int = 1,
    tolerance: float = 1e-12,
) -> OccupationResult:
    """Fill the bands and locate the Fermi level.

    Parameters
    ----------
    eigenvalues
        Shape ``(nspin, nkpoints, nbands)`` in Hartree.
    weights
        k-point weights summing to 1, shape ``(nkpoints,)``.
    nelectrons
        Total electron count in the cell.
    smearing
        Width in Hartree. Ignored for ``fixed``.

    Notes
    -----
    A single Fermi level is used for both spin channels, which is the correct
    equilibrium condition for a collinear magnet free to change its moment.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 3:
        raise ValueError(f"eigenvalues must be (nspin, nk, nbands), got {eigenvalues.shape}")
    weights = np.asarray(weights, dtype=float)
    degeneracy = 2.0 / nspin

    if scheme == "tetrahedron":
        raise NotImplementedError(
            "the linear tetrahedron method lands in Phase 5 with the DOS solver; "
            "use occupations.scheme: gaussian, methfessel-paxton or marzari-vanderbilt"
        )
    if scheme == "fixed" or smearing <= 0.0:
        occupations, fermi, gap = _fill_fixed(eigenvalues, weights, nelectrons, degeneracy)
        entropy_term = 0.0
    else:
        lower = float(eigenvalues.min()) - 20.0 * smearing - 1.0
        upper = float(eigenvalues.max()) + 20.0 * smearing + 1.0

        def electrons(mu: float) -> float:
            f = occupation_function((eigenvalues - mu) / smearing, scheme, order)
            return degeneracy * float(np.einsum("k,skn->", weights, f))

        if electrons(upper) < nelectrons - 1e-9:
            raise ValueError(
                f"only {electrons(upper):.4f} of {nelectrons:g} electrons fit in "
                f"{eigenvalues.shape[2]} bands; raise occupations.nbands"
            )
        for _ in range(200):
            fermi = 0.5 * (lower + upper)
            if electrons(fermi) < nelectrons:
                lower = fermi
            else:
                upper = fermi
            if upper - lower < tolerance:
                break
        fermi = 0.5 * (lower + upper)
        reduced = (eigenvalues - fermi) / smearing
        occupations = occupation_function(reduced, scheme, order)
        entropy = entropy_function(reduced, scheme, order)
        entropy_term = smearing * degeneracy * float(np.einsum("k,skn->", weights, entropy))
        # A gap is a statement about the occupations, not about the smearing scheme: if no
        # state anywhere is partially filled then the Fermi level sits in a gap and the
        # HOMO-LUMO separation is meaningful even though smearing was switched on.
        gap = None
        if float(np.max(np.minimum(occupations, 1.0 - occupations))) < 1e-6:
            filled = eigenvalues[occupations > 0.5]
            empty = eigenvalues[occupations <= 0.5]
            if filled.size and empty.size:
                gap = float(empty.min() - filled.max())

    per_spin = degeneracy * np.einsum("k,skn->s", weights, occupations)
    magnetization = float(per_spin[0] - per_spin[1]) if nspin == 2 else 0.0
    return OccupationResult(
        occupations=occupations,
        fermi_level=float(fermi),
        entropy_term=float(entropy_term),
        gap=None if gap is None or gap <= 0 else float(gap),
        magnetization=magnetization,
    )
