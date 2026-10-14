# Architecture Decision Records

One decision per file, numbered in the order they were made. An ADR records *that* a
decision was made and *why*, so that the next person does not "fix" something that was
deliberate. Only decisions that are hard to reverse, surprising without context, and the
result of a real trade-off get a file.

| ADR | Decision |
|---|---|
| [0001](0001-phase-10-scope-orbital-free-and-reduced-order.md) | Phase 10 scope: orbital-free DFT with non-ML KEDFs as the core, POD-Galerkin reduced Kohn-Sham second, tensor-train limited to a rank survey; Route C/D and nuclear-configuration compression excluded *(KEDF priorities and sequencing amended by 0015, 0016)* |
| [0002](0002-orbital-free-needs-local-pseudopotentials.md) | ~~Orbital-free DFT needs local pseudopotentials: BLPS for real runs, NC-local stub for development, no in-house generation yet~~ **superseded by 0008** |
| [0003](0003-validate-orbital-free-against-our-own-kohn-sham.md) | Validate orbital-free against amaterasu's own Kohn-Sham on the same pseudopotential; literature second; Al then Si-as-failure-study *(element list amended by 0008)* |
| [0004](0004-electrons-theory-config-section.md) | The engine switch is a new `electrons.theory` config section, not `pseudo.method` |
| [0005](0005-ground-state-result-contract.md) | Solvers consume a common `GroundState` contract; eigenvalue data is Kohn-Sham-specific |
| [0006](0006-reduced-order-kohn-sham-does-not-claim-wall-clock-speedup.md) | The reduced-order Kohn-Sham track is gated on accuracy and basis size, not wall-clock: Galerkin projection alone does not speed up `H` application |
| [0007](0007-orbital-free-is-an-sp-element-method.md) | Orbital-free DFT is an sp-element method by physics, not by data; library files bundled unchanged only (CC BY-ND) *(one of its two reasons withdrawn, and M5 retired, by 0008)* |
| [0008](0008-nonlocal-pseudopotentials-via-the-density-matrix.md) | Nonlocal pseudopotentials *are* usable in orbital-free DFT, through a density-matrix functional; PseudoDojo (85 elements) replaces BLPS as the production source, BLPS becomes a zero-projector reference test, M5 is retired; running is not accuracy, so elements carry an evidence class |
| [0009](0009-density-matrix-parameters-are-a-kedf-patch.md) | The density-matrix parameters `A` and `q` patch the kinetic functional rather than describe the pseudopotential: ship `A=0, q=1` uncalibrated, calibrate against the isolated atom, use the published values only to verify our implementation |
| [0010](0010-measure-the-kinetic-energy-kernel.md) | Measure the kinetic energy kernel with our own Kohn-Sham engine (`solver: response`) instead of fitting around it; not the Sternheimer route; XWM deliberately not implemented |
| [0011](0011-neural-networks-for-the-kinetic-functional-but-not-for-exchange-correlation.md) | Neural networks are admitted as an optional kinetic energy density functional (two routes: offline discovery with an analytic product, and an opt-in PyTorch network trained on our own Kohn-Sham data with beryllium held out) but not as an exchange-correlation functional; the analytic ladder stays the default and the ADR-0008 gate table decides. Records the correction that a kinetic functional contributes no explicit force term -- the costs are stress and training data, not forces |
| [0012](0012-evaluation-of-two-dft-acceleration-surveys.md) | Two AI-written DFT acceleration surveys rated on seven axes at three levels and ranked in two tracks: practical (P1 per-stream extrapolation, P2 Harris-Foulkes, P3 LDOS preconditioner, P4 XL-BOMD, P5 preconditioned relaxation) and research (R1 rank-preserving goal-oriented error control first; embedding library, reduced density basis and frozen subspace conditional or held); ADR-0001's exclusions stand, each with a re-evaluation trigger; novelty literature-checked *(B8 promoted, C4 tied to Phase 8, R5/R6 added by 0015, 0016)* |
| [0013](0013-native-solvers-ported-from-akitsushima.md) | Native relaxation, MD, CI-NEB, elastic, TS analysis and phonons are ported from akitsushima by copy (GCMC and Green-Kubo are not), behind one energy/force/stress contract in akitsushima's units with a `context` that names the trajectory stream; ASE stays required and becomes the reference; conformance tests by golden references plus an opt-in live cross-check *(amends the plan's "do not re-implement relax/MD/NEB"; port list and timing amended by 0017)* |
| [0014](0014-output-stress-is-positive-under-compression.md) | All outputs use akitsushima's conventions -- stress positive under compression, eV/Å^3 in data, GPa in logs, MD time in fs -- except the ASE calculator, which keeps ASE's sign; the internal representation is unchanged and the flip happens at the output boundary *(implemented for 0.1.0a3)* |
| [0015](0015-general-purpose-scope-and-standard-feature-roadmap.md) | amaterasu becomes a general-purpose plane-wave code: standard features ordered bulk -> magnetic -> defects -> surfaces, band unfolding for NC and PAW, Phase 8 held, GW/BSE, electron-phonon, TD-OFDFT and implicit solvation recorded as low priority; no competition on Kohn-Sham speed (MPI over k-points brought forward), but orbital-free DFT must beat VASP 6.6.1 above a published atom count N\* while passing ADR-0008's accuracy gate; VASP is also a second validation reference |
| [0016](0016-identity-against-dftpy-error-estimated-multi-level.md) | Against DFTpy, the identity is "error-estimated multi-level" electronic structure on NLPPF ("certified" only where a rigorous bound exists); minimal KEDF set first, full ladder kept; DFTpy as an opt-in test oracle; new research items R6 (Zhou-Wang step reused as an error estimator) and R5 (partial-orbital DFT, measurement study first); O2 held, O3 merged into C5a/R2; novelty literature-checked |
| [0017](0017-ports-from-akitsushima-and-ikazuchix.md) | Phase 5 is built from the akitsushima port, whose list grows to EOS, QHA, Grüneisen and dielectric response; Kumagai-Oba ported from ikazuchix with `delta_v` computed by amaterasu, Freysoldt written new and cross-checked; amaterasu writes an ikazuchix YAML snippet; ikazuchix is not changed |

## Decisions recorded here that did not need their own ADR

These came out of the same design session (2026-09-15) and are worth remembering, but are
easy to reverse or not surprising, so a line each is enough:

- **Sequencing**: Phase 10 starts *after* the remaining standard-DFT work (Phase 5:
  DOS/PDOS, NEB, phonons, DFT+U) is confirmed working. It depends only on the Phase 2
  plane-wave machinery and *could* run in parallel; the user chose to verify the standard
  path first. Roadmap detail lives in the project plan, not here.
- **Where decisions are written**: ADRs in this directory (this file's existence records
  it); the phase-by-phase roadmap stays in the project plan file.
- **`references/`** (the AI-chat exports that motivated Phase 10) is gitignored, like
  `.claude/` and `.github_release/`: development-process material, not the package.
- **Orbital-free implementation choices** (density as `phi^2` with one normalised field;
  KEDF ladder split into milestones M1 semilocal / M2 Wang-Teter / M3 WGC; WGC kernel by
  second-order Taylor expansion about a reference density, tabulated once; forces from M1,
  semilocal stress from M1, nonlocal-kernel stress at M3, all finite-difference checked;
  `basis.ecut` is the cutoff of `phi` and the density gets `4*ecut` as for
  norm-conserving) are standard practice, easy to revise, and recorded in the plan.
- **Round 3 (2026-09-15)**: internal order M0 -> M1 -> rank survey -> M2 -> M3 -> POD;
  the three M0 prerequisites may land during Phase 5 since they are standard-DFT fixes;
  the initial-density fallback is *uniform* (parameter-free); spin-polarised orbital-free
  is a recorded future item ("M4") via exact spin scaling, not part of Phase 10; the
  reference PDFs are read with the locally installed pymupdf, no new dependency. The BLPS
  Library is CC BY-ND 4.0 and covers nine sp elements (Li, Mg, Al, Si, P, Ga, As, In, Sb).
- **Reduced-order track choices** (Gamma-only first, per-k bases later; frozen-phonon Si
  as the parameter family) and the **tensor-train rank survey** (QTT via a self-written
  TT-SVD on power-of-two grids; density, Hartree potential, orbitals and the Wang-Teter
  kernel at relative errors 1e-3/1e-6/1e-9; no new dependency) likewise live in the plan.
- **Nonlocal-pseudopotential session (2026-09-15, Q1-Q16)**: the trigger was a nuclear-physics
  paper (Wu, Colo, Hagino, Zhao, Phys. Rev. Lett. 136, 092501 (2026)) that turned out to be
  Wang-Teter applied to nucleons -- no pseudopotentials, nothing new for this project -- but
  the search it prompted found the electronic-structure work that does matter. Choices too
  small for their own ADR: the double integral is evaluated as a pair sum over density-grid
  points inside each core sphere, with the pair geometry precomputed and the egg-box error
  measured immediately (an atom-centred radial-by-angular grid reusing the PAW quadrature is
  the fallback, and a WGC-style Taylor expansion of the kernel the later acceleration, kept
  alongside the pair sum as its reference implementation); `electrons.nonlocal_pseudo.enabled:
  auto` switches on the presence of projectors, so BLPS files keep working untouched;
  ordering becomes M0 -> M1 -> kernel measurement -> rank survey -> M2 -> nonlocal
  pseudopotential functional -> M3 -> POD.
- **DFT-acceleration review session (2026-09-30, Q1-Q25)**: produced ADR-0012..0014. Too
  small for their own ADR: no genetic-algorithm search history exists, so R1 is validated
  on three structure families built here (ADR-0012); the port and the golden-reference
  generator never write into the akitsushima repository; decisions are written in English
  in the ADRs and in Japanese in the plan; phase numbers 11 and 12 are labels, not an
  execution order, and timing stays with the user.
- **Backlog session (2026-10-15, Q1-Q34)**: produced ADR-0015..0017. Too small for their
  own ADR: GitHub Actions runs **only on `workflow_dispatch`** (lint, ubuntu Python
  3.10-3.13, one macOS 3.13 job, physics as an opt-in input) and is run, like any release,
  only on the user's explicit instruction, to stay within the free tier -- scheduled and
  push-triggered runs had exhausted it by 2026-09-18; 0.1.0a3 is a correctness release
  (configuration keys that are accepted but ignored are rejected at load time,
  `parallel.numba` defaults to false, ADR-0014 applied with forces and stress in HDF5,
  README fixes) with no Phase 5 features; the backlog and the full execution order live
  in the plan, in Japanese.
