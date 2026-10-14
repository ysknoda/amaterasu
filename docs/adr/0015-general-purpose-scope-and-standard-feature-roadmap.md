---
status: accepted
date: 2026-10-15
amends: the plan's framing ("~50-atom research and education use"), ADR-0001 (sequencing), ADR-0012 (scale premise, B8, C4 trigger)
---

# amaterasu becomes a general-purpose plane-wave code with the standard features users expect, ordered bulk -> magnetic -> defects -> surfaces; it does not compete on Kohn-Sham speed, but orbital-free DFT must beat VASP above a published atom count

Until now the plan described amaterasu as a ~50-atom research and education code whose
standard-DFT work stops at Phase 5, with everything after that serving the user's own
methods (orbital-free DFT, R1). On 2026-10-15 the user widened the goal: amaterasu should
be a code that many people can use, as they use Quantum ESPRESSO or VASP -- the standard
features included, easy to use -- while remaining the vehicle for the user's own methods.

That changes how features are judged. Spin-orbit coupling, meta-GGA, hybrids, dispersion
corrections, dipole corrections and charged-cell corrections were out of every roadmap;
they are now real work, and a standard feature is no longer dismissed for not being new.
It also creates a tension the plan already recorded: amaterasu's Kohn-Sham engine is
estimated at 10-100 times slower than VASP. A general-purpose code has to be chosen for
something.

## Decision

**What amaterasu is chosen for.** Not Kohn-Sham speed. The target is 50-100 atoms on a
workstation, and the strengths are being Python-native, integrated with ASE, transparent,
easy to configure, and validated in public. Phase 9's numba and MPI plans stand, with one
exception brought forward: **MPI over k-points**, which is cheap to build and pays most
for the small cells and metals that most users run. Band and plane-wave parallelism and a
Gamma-only real-wavefunction path wait for a Phase 9 profile.

**Orbital-free DFT is where amaterasu must be faster, and the claim is made testable:**

- *Reference*: VASP 6.6.1, Kohn-Sham PBE with PAW at converged settings, on the same
  machine with the same number of MPI ranks.
- *Systems*: fcc Al supercells from 32 to about 2048 atoms.
- *Gate*: there is an atom count N\* such that for every size N >= N\*, amaterasu's
  orbital-free wall time is below VASP's Kohn-Sham wall time; N\* is measured and
  published. "Faster at every size" is not promised, because VASP finishes a 32-atom cell
  in seconds.
- *Accuracy*: the same runs must pass ADR-0008's gate (V0 within 1 %, B0 within 10 %).
- *Configuration*: the production combination -- the nonlocal pseudopotential functional
  (ADR-0008) with Wang-Teter or WGC. Runs with BLPS local pseudopotentials are
  reported for reference only.
- *Consequences*: numba and MPI for the orbital-free engine are brought forward from
  Phase 9; that functional's core-sphere pair sum, which is linear in atoms but quadratic in the
  points inside each sphere and is the largest threat to the gate, gets a cost cap; timing
  starts at M2.

**VASP is also the second validation reference for standard features** -- NiO with +U
(moments and gap), spin-orbit band splittings, HSE gaps, charged-defect corrections --
compared within tolerances, because POTCARs and JTH/PseudoDojo datasets differ. The
primary gate for orbital-free DFT is still amaterasu's own Kohn-Sham (ADR-0003).

**Standard features, by user group, in this order: (a) bulk -> (c) magnetic/correlated ->
(d) defects/semiconductors -> (b) surfaces.**

| Group | Features |
|---|---|
| (a) bulk | DOS/PDOS (tetrahedron), equation of state and elastic constants, phonons with thermodynamics, quasi-harmonic approximation and Grüneisen parameters (ported, ADR-0017), CI-NEB, DFT+U, `amaterasu plot`, CUBE/XSF density and potential output; then Phase 6 DFPT for electric-field and phonon perturbations (Born charges and epsilon-infinity for the LO-TO splitting). Strain DFPT later -- elastic constants come from finite differences meanwhile |
| (c) magnetic | spin-orbit coupling and noncollinear magnetism (fully relativistic pseudopotentials, spinors), constrained moments (fixed spin moment), r²SCAN, Wannier90 file output (Phase 7's first stage). Native MLWF held |
| (d) defects | band unfolding (first in this group), charged-cell corrections (Kumagai-Oba ported, Freysoldt new; ADR-0017), ultrasoft pseudopotentials and mixed NC/PAW cells, HSE06 with ACE, optical dielectric function (independent-particle, with the PAW correction), Berry-phase polarisation, effective masses |
| (b) surfaces | dipole correction / ESM, D3 dispersion, work function, in-plane band unfolding for slabs |

**Band unfolding** covers norm-conserving and PAW -- with the projector (S-metric)
correction, since pseudo-wavefunctions alone give weights that do not sum to one -- and
symmetry averaging over equivalent paths (Popescu and Zunger, PRB 85, 085201 (2012); Ku,
Berlijn and Lee, PRL 104, 216401 (2010); BandUP, Medeiros et al., PRB 89, 041407 (2014)).
Its gates are exact: a perfect-crystal supercell unfolds to weights of 0 or 1 that
reproduce the primitive bands, and every state's weights sum to one.

**Usability work, early**: a documentation site with an input reference generated from
`config/schema.py` and API documentation alongside it; CUBE/XSF output and pymatgen
interoperability; more families in `fetch-pseudos`. VASP/QE input converters come later.
Publishing to PyPI happens at beta and only on the user's instruction.

**Sequencing.** After Phase 5 the standard track and the research track alternate, items
that serve both going first (the ported solvers, MPI over k-points). This replaces the
plan's "research after Phase 5" and ADR-0001's sequencing note. The full order is:

1. 0.1.0a3, a correctness release with no Phase 5 features: load-time rejection of every
   configuration key that is accepted but ignored (ADR-0004's rule, previously broken by
   `system.hubbard`, `basis.type: mixed`, `parallel.*`, `output.checkpoint_every` and
   `--restart` outside `solver: scf`), moving mid-run refusals to load time,
   implementing `output.log_level`, applying ADR-0014 with forces and stress in HDF5, and
   README fixes.
2. Phase 5: the energy/force/stress contract and adapter of ADR-0013 -> DOS/PDOS -> EOS
   and elastic constants -> phonons, thermodynamics, QHA -> NEB -> DFT+U -> plotting and
   CUBE/XSF -> Phase 6 electric-field and phonon DFPT. MPI over k-points lands in this
   period.
3. Alternating milestones: the standard groups (c), (d), (b) as tabled above, and the
   research order of ADR-0016.

**Phases 6-8.** Phase 6 as above. Phase 7 is reduced to Wannier90 output for now. Phase 8
(mixed basis) is **held**: its purpose was 3d/4f elements at low cutoff, and ultrasoft
pseudopotentials and PAW now cover that.

**Recorded, low priority -- under consideration, not planned**: GW/BSE, electron-phonon
coupling, time-dependent orbital-free DFT, implicit solvation (to be revisited after ESM
in group (b)).

## Consequences for earlier records

- **ADR-0012** judged effects at "Python, ~50 atoms". Its ratings are not redone now;
  the premise change is handled by its existing triggers. B8 (ACE) becomes P1 now that
  HSE06 is in group (d). C4's trigger ("Phase 8 failing its gate") cannot fire while
  Phase 8 is held, so C4 is re-evaluated together with Phase 8.
- **ADR-0001**'s note that Phase 10 starts after Phase 5 stands; what follows Phase 5 is
  now the alternation above.

## Re-evaluation triggers

- Phase 8 (and with it ADR-0012's C4): ultrasoft or PAW datasets turning out to need
  impractical cutoffs for 3d/4f elements.
- Band/plane-wave MPI, Gamma-only path, GPU: the Phase 9 profile.
- Native MLWF: a user need that Wannier90 output does not meet.
- The low-priority list: an explicit user decision.
