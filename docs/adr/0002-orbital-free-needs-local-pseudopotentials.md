---
status: accepted
date: 2026-09-15
---

# Orbital-free DFT uses local pseudopotentials only: BLPS for real runs, the local part of a norm-conserving file as a development stub, and no in-house generation

Orbital-free DFT has no orbitals, so a Kleinman-Bylander nonlocal projector -- which acts
on orbitals -- has no meaning in it. Neither the PseudoDojo norm-conserving files nor the
JTH PAW datasets this code already reads can be used as they are. None of the eight
survey documents that motivated Phase 10 mention this; it is the single largest practical
constraint on the work and it changes what has to be sourced.

Decision:

- **Production**: bulk-derived local pseudopotentials (BLPS) from the Carter group (Zhou,
  Wang, Carter, Phys. Rev. B 69, 125109 (2004) and successors), for Al and Si first. The
  files and the papers are obtained by the user and quoted with provenance, per the
  project rule that reference data is not reconstructed from memory.
- **Development stub**: the local part of an existing norm-conserving file, with the
  nonlocal projectors dropped. This is physically wrong -- it is not a complete
  pseudopotential -- and exists only so the minimiser, the functionals and the tests can
  be built before the BLPS files arrive. Any run using it must say so in its log.
- **Not now**: fitting our own local pseudopotential to Kohn-Sham band structures. That is
  a small project of its own and is recorded as a later option, not started.

Consequence: `electrons.theory: orbital-free` must refuse a dataset that carries nonlocal
projectors or PAW augmentation unless the stub mode is explicitly requested, rather than
silently using its local part.

## What the BLPS files turned out to be (measured 2026-09-15)

The Princeton BLPS Library ships each potential twice: `*.psp` (real space) and
`*.recpot` (reciprocal space, CASTEP convention). The `.psp` files are **ABINIT psp8
format with zero nonlocal projectors** (`pspcode 8`, `pspxc 11` = PBE, `nproj 0 0 0 0 0`,
1601 points on a uniform 0.01 Bohr grid to 16 Bohr, Coulomb tail exactly `-Z_ion/r`
at the last point). amaterasu's existing psp8 reader reads `al.gga.psp` and `si.gga.psp`
unchanged, and a Kohn-Sham SCF on the resulting zero-projector pseudopotential converges
(Al fcc, 11 iterations, nonlocal energy identically zero). So:

- **`.psp` via the existing reader is the format; `.recpot` is not needed** and gets no
  reader. The same file feeds both engines, which is exactly what ADR-0003's comparison
  requires.
- Three small prerequisites surfaced, all in the standard-DFT path and none specific to
  orbital-free theory:
  1. the initial density needs a fallback (superposed normalised Gaussians, or uniform)
     for a species whose file carries no atomic valence density -- psp8 BLPS files carry
     none, and `initial_density()` currently raises;
  2. the psp8 reader takes the element symbol from the title line, which here reads
     `al_locpsp` / `Si_locpsp`; it must fall back to the symbol implied by `zatom`, or
     the registry's symbol check rejects the file;
  3. `check()` warns that `V_loc` departs from `-Z_ion/r` beyond 3 Bohr; BLPS potentials
     become Coulombic only from 4-5 Bohr (stated in their title line), so the check's
     radius must be a property of the file, not a constant.
