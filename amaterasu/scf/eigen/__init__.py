"""Iterative eigensolvers for the Kohn-Sham equations at a fixed potential."""

from __future__ import annotations

from .davidson import DavidsonResult, davidson, generalized_davidson
from .precond import teter_payne_allan
from .subspace import orthogonalize_against, orthonormalize, random_bands, rayleigh_ritz

__all__ = [
    "DavidsonResult",
    "available_eigensolvers",
    "davidson",
    "generalized_davidson",
    "orthogonalize_against",
    "orthonormalize",
    "random_bands",
    "rayleigh_ritz",
    "solve",
    "teter_payne_allan",
]

#: Eigensolvers implemented so far. ``lobpcg`` and ``rmm-diis`` are accepted by the input
#: schema but land later; they are alternatives to Davidson, not prerequisites for it.
_IMPLEMENTED = ("davidson",)


def available_eigensolvers() -> tuple[str, ...]:
    return _IMPLEMENTED


def solve(
    name: str, apply_hamiltonian, guess, kinetic, *, apply_overlap=None, **kwargs
) -> DavidsonResult:
    """Dispatch to the eigensolver named in ``scf.eigensolver``.

    ``apply_overlap`` switches to the generalised problem ``H c = eps S c``, which is what
    a PAW calculation needs; a norm-conserving one leaves it ``None``.
    """
    if name != "davidson":
        raise NotImplementedError(
            f"eigensolver {name!r} is not implemented yet; use "
            f"{' or '.join(repr(x) for x in _IMPLEMENTED)} (scf.eigensolver)"
        )
    if apply_overlap is None:
        return davidson(apply_hamiltonian, guess, kinetic, **kwargs)
    return generalized_davidson(apply_hamiltonian, apply_overlap, guess, kinetic, **kwargs)
