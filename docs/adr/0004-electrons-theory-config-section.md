---
status: accepted
date: 2026-09-15
---

# The electronic-structure engine is chosen by a new top-level `electrons` config section, not by `pseudo.method`

Orbital-free DFT is not a pseudopotential method; it replaces the Kohn-Sham engine
underneath the solvers, and `solver: scf | relax | md` should keep working on top of it
unchanged. Putting the switch under `pseudo.method` (next to `nc` and `paw`) would make
it look like a third kind of dataset and would invent false exclusions such as "orbital-
free is incompatible with PAW" -- true, but for the reason that OF cannot use nonlocal
datasets at all (ADR-0002), not because it is a sibling of PAW.

Decision: a new section

```yaml
electrons:
  theory: kohn-sham        # default; every existing input is unchanged
  # theory: orbital-free
  kedf: {name: wgc, ...}   # orbital-free only
  reduced_basis: {...}     # the POD-Galerkin track, Kohn-Sham only
```

`theory` defaults to `kohn-sham`, so no existing input file changes meaning. Validation
rejects combinations that are not implemented (a `kedf` under `kohn-sham`; spin
polarisation under `orbital-free`) at load time, in the same one-pass error report the
rest of the schema uses.

Spin polarisation under `orbital-free` is rejected by *scope*, not by theory. The
non-interacting kinetic energy obeys the exact spin-scaling relation
`T_s[n_up, n_dn] = T_s[2 n_up]/2 + T_s[2 n_dn]/2` (Oliver and Perdew, Phys. Rev. A 20,
397 (1979); to be confirmed against the source before it is coded), so every
unpolarised KEDF defines a polarised one and the orbital-free minimiser generalises to two
fields with two constraints. It is left out of Phase 10 to keep the first minimiser to one
field, and because the materials where spin matters most -- Fe, Co, Ni -- are d-metals
that local pseudopotentials and orbital-free theory both describe poorly, so the payoff
is small. The error message must say "not implemented", never "not possible".

Consequence for the code: the solvers and the ASE calculator currently consume
`SCFResult` fields that orbital-free DFT does not produce -- eigenvalues, occupations,
Fermi level, gap. A common ground-state result contract that both engines satisfy is
needed before `relax`/`md` can run on an orbital-free engine; that design is a Phase 10
task, and the shape of it is an open question recorded in the plan.
