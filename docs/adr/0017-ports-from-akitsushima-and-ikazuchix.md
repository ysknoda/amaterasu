---
status: accepted
date: 2026-10-15
amends: ADR-0013 (port list and timing), the plan's Phase 5 (NEB and phonons through ASE/phonopy)
---

# Phase 5 is built from the akitsushima port, whose list grows to bulk thermodynamics and dielectric response; the Kumagai-Oba charge correction is ported from ikazuchix, Freysoldt is written new, and ikazuchix is not changed

ADR-0013 decided to port akitsushima's native solvers by copy, as a later Phase 11, with
Phase 5 still built on ASE and phonopy. Two things changed. ADR-0015 made bulk properties
the first user group, and since 2026-09-30 akitsushima has gained exactly those
properties: an elastic-constants command with polycrystalline averages, internal strain
and equation-of-state fits (akitsushima ADR-0017), phonon thermodynamics, Grüneisen
parameters and the quasi-harmonic approximation (its ADR-0018), and non-analytic
correction from input Born charges, dielectric response and piezoelectricity (its
ADR-0019). Building Phase 5 on ASE/phonopy and then porting would build the user-facing
path twice. And ADR-0015 added charged-defect corrections, which the user's other sister
package, ikazuchix (`~/Claude_Code/defect_formation_energy/ikazuchix`, MIT, about 4,000
lines, numpy and PyYAML only), already implements in part.

## Decision

**akitsushima (`~/Claude_Code/akitsushima/akitsushima`, MIT, v0.2.0a13 when surveyed).**

- ADR-0013's method stands unchanged: copy, not dependency; one energy/force/stress
  contract with `context`; one unit and sign adapter guarded by finite differences;
  `PORTED_FROM.md` and per-file source headers; golden-reference conformance tests plus
  the opt-in `pytest -m akitsushima`; nothing is written into the akitsushima repository.
- **The port list grows**: relaxation, MD, CI-NEB, elastic constants with the
  equation-of-state fit, phonons with thermodynamics, QHA and Grüneisen parameters,
  non-analytic correction and dielectric/piezoelectric response, transition-state
  analysis. GCMC and Green-Kubo remain unported, for ADR-0013's reasons.
  `engine/coulomb/esm.py` is a point-charge Poisson solver; it is a starting point for
  ESM in group (b) of ADR-0015, not a direct port, because the DFT Hartree term needs it
  rebuilt.
- **Timing**: the port is no longer a later Phase 11. Phase 5 starts with the contract and
  adapter, and its NEB, phonons, EOS, elastic constants and QHA are the ported versions
  from the start. For these Phase 5 features ASE's NEB and phonopy are used **inside the
  tests only**, as the independent reference implementations ADR-0013 calls for, so
  there is one user-facing path, not two. The ASE relaxation and MD that already ship
  (Phase 3) stay as they are, under ADR-0013's rules, until the native ones pass its
  gate.
- Until Phase 6 DFPT exists, the non-analytic correction takes Born charges and
  epsilon-infinity as user input, as akitsushima already does.

**ikazuchix.**

- It implements the Kumagai-Oba (KO) correction (Kumagai and Oba, PRB 89, 195205 (2014))
  only. Freysoldt (FNV; Freysoldt, Neugebauer and Van de Walle, PRL 102, 016402 (2009)) is
  **not** implemented there -- it is cited, nothing more.
- **Port the KO core by copy**: the anisotropic point-charge Ewald sum, the lattice energy
  and the point-charge potential (`ewald.py` and `ewald_kernels.py`, about 720 lines,
  `KumagaiObaCorrectionCalculator` and `point_charge_potential`), with the same
  provenance rules as the akitsushima port.
- **Remove the manual step.** In ikazuchix the alignment term `delta_v` is averaged by
  hand from VASP OUTCAR site potentials and typed into YAML, and the dielectric tensor is
  an input. amaterasu instead writes atomic-site potentials and planar-averaged
  potentials, takes the dielectric tensor from Phase 6 DFPT (electronic plus ionic), and
  computes `delta_v` itself.
- **Implement FNV new in amaterasu**, cross-checked against KO, which must agree with it
  in the isotropic limit.
- **ikazuchix is not changed.** Formation energies, transition levels and defect
  equilibria stay there: they are code-independent post-processing, and keeping them
  there keeps ikazuchix useful to VASP users. amaterasu writes a snippet in ikazuchix's
  existing YAML input format, so nothing is transcribed by hand. ikazuchix reading
  amaterasu's output files was considered and is not wanted.
- The charged-defect corrections are also one of the comparisons against VASP that
  ADR-0015 lists.

## Licence

Both sister packages are MIT and by the same author as amaterasu (Apache-2.0), so the
ports are unencumbered; the per-file source headers are kept for provenance.
