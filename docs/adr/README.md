# Architecture Decision Records

One decision per file, numbered in the order they were made. An ADR records *that* a
decision was made and *why*, so that the next person does not "fix" something that was
deliberate. Only decisions that are hard to reverse, surprising without context, and the
result of a real trade-off get a file.

| ADR | Decision |
|---|---|
| [0001](0001-phase-10-scope-orbital-free-and-reduced-order.md) | Phase 10 scope: orbital-free DFT with non-ML KEDFs as the core, POD-Galerkin reduced Kohn-Sham second, tensor-train limited to a rank survey; Route C/D and nuclear-configuration compression excluded |
| [0002](0002-orbital-free-needs-local-pseudopotentials.md) | Orbital-free DFT needs local pseudopotentials: BLPS for real runs, NC-local stub for development, no in-house generation yet |
| [0003](0003-validate-orbital-free-against-our-own-kohn-sham.md) | Validate orbital-free against amaterasu's own Kohn-Sham on the same pseudopotential; literature second; Al then Si-as-failure-study |
| [0004](0004-electrons-theory-config-section.md) | The engine switch is a new `electrons.theory` config section, not `pseudo.method` |
| [0005](0005-ground-state-result-contract.md) | Solvers consume a common `GroundState` contract; eigenvalue data is Kohn-Sham-specific |
| [0006](0006-reduced-order-kohn-sham-does-not-claim-wall-clock-speedup.md) | The reduced-order Kohn-Sham track is gated on accuracy and basis size, not wall-clock: Galerkin projection alone does not speed up `H` application |

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
- **Reduced-order track choices** (Gamma-only first, per-k bases later; frozen-phonon Si
  as the parameter family) and the **tensor-train rank survey** (QTT via a self-written
  TT-SVD on power-of-two grids; density, Hartree potential, orbitals and the Wang-Teter
  kernel at relative errors 1e-3/1e-6/1e-9; no new dependency) likewise live in the plan.
