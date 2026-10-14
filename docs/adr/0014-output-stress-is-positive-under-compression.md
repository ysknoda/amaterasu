---
status: accepted (implemented 2026-10-15, for 0.1.0a3)
date: 2026-09-30
---

# Every amaterasu output uses akitsushima's conventions -- stress positive under compression, in eV/Å^3 -- except the ASE calculator, which must keep ASE's

amaterasu and its sister package akitsushima exchange energies, forces and stresses
(ADR-0013), and they disagreed on the stress sign: amaterasu followed ASE (`sigma =
(1/Omega) dE/d eps`, positive under tension), akitsushima uses `sigma = -(1/V) dE/dT` for
`lattice' = lattice @ T`, positive under compression. One convention across both packages
removes the sign flip that would otherwise sit at every boundary between them.
akitsushima's ADR-0001 records why that matters: the wrong sign in one pressure expression
was shipped once, and made its MTTK barostat drive a compressed cell to collapse further
and an expanded one to expand further.

To a reader used to ASE or VASP, a positive stress on a compressed cell looks like a bug.
It is deliberate.

## Decision

- **All outputs follow akitsushima**: energy eV, length Å, force eV/Å, stress eV/Å^3 in
  data (HDF5, EFS files, `force_fn` results), GPa in human-readable logs and in input
  thresholds (`pmax`, `pressure`), stress positive under compression. MD time is output
  in fs; wall-clock time stays in s.
- **The ASE calculator is the one exception.** `Amaterasu.get_stress()` keeps ASE's sign,
  because ASE's cell filters and optimisers assume it, and the ASE route is the reference
  the native solvers are gated against (ADR-0013). Flipping it would break that reference.
- **The internal representation does not change.** `properties/stress.py` computes `sigma
  = (1/Omega) dE/d eps` in Hartree atomic units and is finite-difference verified. The
  sign is flipped at the output boundary, the same place units are converted, following
  amaterasu's rule of converting once at the boundary. For a symmetric stress tensor the
  two conventions differ by sign alone.
- **Pressure is unchanged.** It is a physical quantity: `P = -tr(sigma_ASE)/3 =
  +tr(sigma)/3`, positive when the cell is compressed, in both conventions.
- **Files say which convention they use.** HDF5 output gains an attribute
  `stress_convention = "positive_under_compression"` and a format-version bump.

## Why now

amaterasu does not yet write forces or stress to its HDF5 output; stress leaves the code
only through the ASE calculator and as a pressure in relax/MD logs. No existing output
file changes meaning, so this is the cheapest moment to fix the convention. The README's
units table ("pressure, stress: GPa"; "time: s") changes when this is implemented, not
before, because it describes what the code does today.
