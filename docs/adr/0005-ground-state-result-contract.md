---
status: accepted
date: 2026-09-15
---

# Solvers consume a common ground-state result contract; eigenvalues, occupations, Fermi level and gap are Kohn-Sham-specific and are not part of it

Today `SCFResult` is read directly by the ASE calculator, `solvers/scf.py`,
`solvers/bands.py` and `io/results.py`, and four of the fields they read -- eigenvalues,
occupations, Fermi level, band gap -- do not exist in orbital-free theory. There is no
orbital to have an eigenvalue. The tempting fix, an `SCFResult` with those fields set to
`None`, would scatter `if eigenvalues is not None` through every consumer and let
`solver: bands` appear to run on an orbital-free engine and fail late.

Decision: a small `GroundState` protocol holding only what every engine produces --
energies, density, convergence flag, iteration count, wall time -- is what `relax`, `md`,
the calculator's energy/force path and the HDF5 writer's common part consume. The
Kohn-Sham engine returns an `SCFResult` that satisfies it and additionally carries the
eigenvalue data; the orbital-free engine returns an `OrbitalFreeResult` that satisfies
only the protocol. Anything that needs eigenvalues (`bands`, the eigenvalue table in the
SCF report, the `scf/` group in the HDF5 file) asks for the Kohn-Sham result explicitly
and is refused, at input validation, under `electrons.theory: orbital-free`.

Consequence: PAW forces read the occupancy matrices from the Kohn-Sham result; that path
stays Kohn-Sham-only, which is consistent, since orbital-free theory cannot use a PAW
dataset at all (ADR-0002).
