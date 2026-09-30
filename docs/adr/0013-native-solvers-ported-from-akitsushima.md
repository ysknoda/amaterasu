---
status: accepted
date: 2026-09-30
amends: the plan's settled decision 2 ("ASE is required; relax, MD and NEB are not re-implemented")
---

# Native relaxation, MD and NEB are ported from akitsushima by copy, behind one energy/force/stress contract that also carries trajectory context; ASE stays, as the reference

The plan settled early that ASE is a required dependency and that amaterasu drives ASE's
optimisers, dynamics and NEB rather than re-implementing them. That kept Phases 3-5 small,
but it caps what the electronic solver can do along a trajectory: an ASE `Calculator` sees
one structure at a time, so it cannot tell a molecular-dynamics step from an NEB image
from a phonon displacement, and XL-BOMD (ADR-0012, P4) cannot be built at all, because it
integrates an auxiliary density *inside* the MD integrator.

The user's sister package akitsushima (`/Users/yusuke/Claude_Code/akitsushima`, same
author, MIT) already has a relaxation with six algorithms, NVE/NVT/NPT MD, CI-NEB, a
transition-state analyser, finite-displacement phonons, elastic constants, GCMC and
Green-Kubo thermal conductivity, all driven through one callable. amaterasu gets its native
solvers by porting those.

## Decision

**Port by copy, not by dependency.** The solvers are copied into amaterasu and stripped of
their akitsushima-only imports (neighbour lists, MLIP checks, POSCAR/XDATCAR writers). A
dependency, or registering amaterasu as an akitsushima engine, would keep one copy, but the
reason for porting is to diverge: context-aware extrapolation, XL-BOMD in the integrator,
SCF tolerance tied to force error. Those are DFT-specific changes that do not belong in a
classical force-field package. Drift between the copies is caught by the conformance tests
below, and `amaterasu/solvers/PORTED_FROM.md` records, per module, the akitsushima commit
it came from and every deliberate change since. Each ported file names its source file
and commit in its header.

**ASE stays a required dependency and becomes the reference.** The ASE route of Phases 3-5
(ASE optimisers, ASE CI-NEB, phonopy) is not removed. A native solver passes its gate when
it matches the ASE route on the same system and settings -- the same pattern as DFPT
against finite displacement (Phase 6) and native MLWF against Wannier90 (Phase 7).

**One contract between solver and engine, in akitsushima's conventions.**

    force_fn(lattice, positions, atom_types, neigh_list=None, context=None)
        -> result with .energy, .forces, .stress (and unpacks as (E, F, S))

- Energy in eV, forces in eV/Å, stress in eV/Å^3, **positive under compression**,
  `sigma = -(1/V) dE/dT` for `lattice' = lattice @ T` (akitsushima ADR-0001; amaterasu
  ADR-0014).
- The energy is the **force-consistent** one: with smearing, the free energy `E - TS`
  whose derivatives the forces are. The `sigma -> 0` extrapolated `E0` is reported, never
  passed to a solver.
- amaterasu computes in Hartree atomic units with ASE's stress sign internally; **exactly
  one adapter** converts units and flips the sign, and it is guarded by finite-difference
  tests of `F = -dE/dR` and `sigma = -(1/V) dE/dT` through the adapter.
- `context` names the **stream** the call belongs to (an MD trajectory, NEB image *i*, the
  reference structure of a phonon displacement), the step index and, for dynamics, `dt`.
  The engine keeps extrapolation history per stream, which is what makes ADR-0012's P1
  correct for NEB and phonons, where consecutive calls belong to different streams.
- The result also carries SCF diagnostics -- iterations, final residual, an estimate of
  the force error -- so a solver can tighten or relax the SCF tolerance.
- Callables that do not accept `context` keep working, detected by signature inspection
  exactly as akitsushima already does for `neigh_list`; plain `(E, F, S)` tuples are
  accepted.

**What is ported, in order.**

| Solver | Decision | Why |
|---|---|---|
| Relaxation (six algorithms) | port first | carries P1 and P5 |
| MD, NVE/NVT/NPT | port second | carries P1 and P4 (XL-BOMD). NPT must state how the plane-wave basis follows the cell (constant cutoff vs constant basis) and handle the Pulay stress that comes with it |
| CI-NEB | port third | first real test of per-image streams |
| Elastic constants | port | the cleanest check of the stress convention; cross-checks strain DFPT (Phase 6) |
| Transition-state analysis | port, later | its finite-difference Hessian costs 2(3N+9) DFT calls; pair it with P5 or the Phase 6 analytic Hessian |
| Phonons (finite displacement) | port, low priority | overlaps the phonopy route; kept as a third reference |
| GCMC | **not ported**, recorded | one DFT per trial move, and insertion changes the electron count; akitsushima itself only warns for external engines |
| Green-Kubo thermal conductivity | **not ported as is**, recorded | needs per-atom energies, which DFT does not define uniquely; ab initio Green-Kubo needs the energy-flux theory of Marcolongo, Umari and Baroni, Nat. Phys. 12, 80 (2016), which is a project of its own |

**Conformance tests.** In amaterasu's own `tests/conformance/`, not a third repository
that neither test suite would run.

- *Golden references, always run*: data generated from a pinned akitsushima commit (npz
  plus a provenance JSON) is committed to amaterasu, so amaterasu's suite runs without
  akitsushima installed. The generator, `tests/conformance/generate_from_akitsushima.py`,
  lives in amaterasu and imports akitsushima; nothing is written into the akitsushima
  repository.
- *Live cross-check, opt-in*: `pytest -m akitsushima` runs both packages side by side when
  akitsushima is importable. A failure there after an akitsushima bug fix is the signal to
  read `PORTED_FROM.md` and decide whether to port the fix.
- *Test potential*: Lennard-Jones and Morse `force_fn`s in plain numpy, test-only, so the
  whole suite runs in seconds with no DFT.
- *What is compared*: relaxation -- final structure, energy and step count for each
  algorithm (1e-10); NVE -- positions and velocities over the first 100 steps (1e-10),
  energy-drift statistics only beyond that, because a chaotic trajectory separates from
  rounding differences exponentially; NVT/NPT -- trajectories if the random-number seed
  and call order can be matched, statistics otherwise; CI-NEB -- barrier and image
  energies; elastic -- `C_ij`.

**Where the work happens.** The port is Phase 11, run in a fresh session on a git worktree
of amaterasu (about 5,000 lines; it should not be mixed with Phase 5 work). No akitsushima
session is needed.

## Licence

akitsushima is MIT and amaterasu Apache-2.0. The author is the same person, so the port is
unencumbered; the per-file source header is kept for provenance, not as a licence
requirement.
