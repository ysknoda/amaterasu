---
status: accepted
date: 2026-09-30
---

# Two AI-written DFT acceleration surveys, rated and ranked: a practical track built on trajectory history, and one research item worth doing first -- rank-preserving goal-oriented error control

Two surveys under `references/DFT_acceleration_ideas/` (gitignored, like the rest of
`references/`) generalise the orbital-free idea -- "drop a degree of freedom the answer
does not need" -- to the whole Kohn-Sham pipeline. `DFT高速化_重要要素削減アイデア.md` (ChatGPT,
cited below as **G§n**) is a taxonomy of about seventeen ideas with no minimum experiments
and no risks. `dft_acceleration_ideas.md` (Claude, cited as **C§n** for existing methods
and **C#n** for its eight new proposals) gives each proposal an experiment and a risk, but
assumes VASP and a genetic-algorithm structure search, neither of which is amaterasu.

This record rates every idea from both on the same scale, says which are built, and in
what order. It exists so that the rejected ones are not re-proposed and the accepted ones
are not re-argued. Implementation timing is the user's and is not decided here.

## How the ratings were made

- **One merged list, three groups.** Duplicates across the two surveys are one row with
  both sources. **A** = already decided by ADR-0001..0011 (rated for comparison, not
  re-decided). **B** = established techniques amaterasu does not have yet -- the
  *practical track*. **C** = research ideas -- the *research track*. G§20 (information-
  theoretic DFT) and G§23 (the "Goal-Oriented + Adaptive + Low-Rank + Latent" synthesis)
  are framing, not implementable units, and are not rated.
- **Seven axes, three levels (L/M/H).** Difficulty, expected effect, academic impact,
  novelty, risk, verifiability (can our own Kohn-Sham run be the reference, ADR-0003),
  reuse of existing machinery. Difficulty and risk: H is bad. Three levels, not five,
  because neither survey supports finer numbers; every speed-up figure in them is a guess.
- **No weighted sum.** Verdicts are staged: conflicts with an ADR -> verifiable? ->
  (effect + impact) / difficulty -> risk breaks ties. The two tracks are ranked
  separately, because a technique with no novelty that certainly speeds up MD and a novel
  method that may not are not competing for the same thing.
- **ADR-0001's exclusions stand.** Excluded ideas are rated and stay rejected, each with a
  re-evaluation trigger. ADR-0011 overturned an exclusion because a new fact appeared
  (the kinetic functional became the only error source); neither survey supplies one.
- **Novelty of the research track was checked, not assumed.** Four literature searches
  (C1, C2, C3, C5) were run on 2026-09-30. They were web searches read largely through
  summaries; every verdict below is marked (U) where confidence is under 90%.
- **Effect is judged at amaterasu's scale** (Python, ~50 atoms, FFT- and BLAS-bound). A
  method whose payoff starts at hundreds of atoms can still be admitted to the research
  track, gated on accuracy and scaling rather than wall-clock (the ADR-0006 precedent),
  but the record must say that amaterasu will not be faster for it.

## Ratings

Axes: Dif = difficulty, Eff = expected effect, Imp = academic impact, Nov = novelty,
Rsk = risk, Ver = verifiability, Reu = reuse.

### A -- already decided (not re-decided)

| ID | Idea (source) | Removes | Dif | Eff | Imp | Nov | Rsk | Ver | Reu | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Orbital-free, KEDF ladder + nonlocal pseudopotential functional (G§3, C§2A) | orbitals | M | H | M | L | M | H | H | decided, ADR-0001/0008 |
| A2 | Analytic KEDF from the measured kinetic kernel (C#2) | orbitals (accuracy) | M | M | H | M | M | H | H | decided, ADR-0010/0011 |
| A3 | POD reduced-order Kohn-Sham (C#4, G§3 "orbital compression") | eigenproblem size | M | L | M | M | M | H | H | decided, ADR-0006 |
| A4 | Neural KEDF (C§2A) | orbitals | H | M | M | L | H | H | M | decided, ADR-0011 |

### B -- practical track

| ID | Idea (source) | Removes | Dif | Eff | Imp | Nov | Rsk | Ver | Reu | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Density and wavefunction extrapolation along a trajectory (ASPC; atomic-density-difference extrapolation) (G§9, C§2D) | SCF iterations | L | M | L | L | L | H | H | **adopt, P1** |
| B3 | Harris-Foulkes non-self-consistent energy (C§2) | the SCF itself | L | M | L | L | L | H | H | **adopt, P2** |
| B5 | LDOS preconditioner, Herbst-Levitt (C§2D) | SCF iterations in inhomogeneous systems | M | M | L | L | L | H | H | **adopt, P3** |
| B2 | Extended-Lagrangian BOMD, XL-BOMD (C§2D) | SCF iterations in MD | M | H | L | L | M | H | M | **adopt, P4** (needs native MD) |
| B6 | Preconditioned / low-rank-Hessian relaxation (G§15) | ionic steps | M | M | L | L | L | H | M | **adopt, P5** (needs native relax/TSA) |
| B4 | Chebyshev-filtered subspace iteration (C§2B) | diagonalisation | M | L | L | L | M | H | H | hold: decide from the Phase 9 profile; in Python the FFT dominates |
| B7 | Adaptive k-point sampling (G§10) | k-points | H | M | M | M | M | H | M | hold: pays off for metals only |
| B8 | ACE / ISDF compression of exact exchange (C§2B) | exact exchange | -- | -- | -- | -- | -- | -- | -- | conditional: becomes P1 the day hybrid functionals are added |

### C -- research track

| ID | Idea (source) | Removes | Dif | Eff | Imp | Nov | Rsk | Ver | Reu | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| C2+C6 | **Rank-preserving goal-oriented error control**: a posteriori error of the energy *ranking* of candidate structures, driving Ecut, k-mesh and SCF tolerance (G§14/§22B; C#6's rank metrics) | over-converged discretisation | H | M | H | M-H (U) | M | H | M | **adopt, R1** |
| C5 | Kohn-Sham-in-orbital-free embedding (a) + a descriptor-indexed library of KS-minus-OF corrections (b) (G§17/§11, C#3) | KS for most atoms | H | H | H | a: L, b: M (U) | H | M | M | conditional, **R2** = (b) only; (a) is an application of the orbital-free engine |
| C1 | Orbital-free DFT over an atom-centred reduced density basis (G§19/§3/§22A, C#1) | grid degrees of freedom of the density | H | L | M | L-M (U) | H | H | M | conditional, **R3** |
| C3 | Frozen deep occupied subspace across ionic steps (G§4, C#5) | updates of deep bands | M | L | L | M (U) | M | H | H | hold, **R4** |
| C9b | Learned initial density, followed by a full SCF (initial-guess use of G§8/§13) | SCF iterations | M | L | L | L | L | H | M | hold: not excluded by ADR-0001 (the SCF, not the model, produces the answer), but B1 captures most of the gain |
| C4 | Spatially adaptive resolution (G§5/§11/§16/§22C) | a uniform grid | H | M | M | L | H | M | L | reject: incompatible with a plane-wave basis; the nearest thing is Phase 8's mixed basis |
| C8 | Correlated-sampling stochastic DFT (C#8) | statistical error | H | L | M | M | H | M | M | reject: pays off at thousands of atoms, and at ~50 has no accuracy question worth asking either |
| C7 | Relaxation in latent/collective coordinates (C#7, G§⑦) | ionic steps | -- | -- | -- | -- | -- | -- | -- | reject: ADR-0001 (nuclear-configuration compression) |
| C9a | Learning the SCF map itself, "ML-SCF" (G§8) | SCF iterations | -- | -- | -- | -- | -- | -- | -- | reject: ADR-0001 (a regression replaces the physics) |
| C10 | Local-environment density prediction used as the answer (G§13) | the SCF | -- | -- | -- | -- | -- | -- | -- | reject: ADR-0001 |
| C11 | Cheap DFT + ML correction, delta learning (G§18) | accurate DFT | -- | -- | -- | -- | -- | -- | -- | reject: ADR-0001 |
| C12 | Sparse / low-rank Hamiltonian and density matrix (G§6/§7) | matrix elements | -- | -- | -- | -- | -- | -- | -- | reject: ADR-0001 Route C; thresholding is meaningless for a matrix-free plane-wave `H` |
| C13 | Screened electron-electron interaction (G§12) | long-range interaction | -- | -- | -- | -- | -- | -- | -- | reject: with semilocal XC, Hartree is one FFT and there is nothing to save; revisit with hybrids |

## What both surveys missed, and what it changed

- **Fewer variables is not less work (C1).** Expanding the density in ~10 functions per
  atom leaves every energy and gradient evaluation on the full grid: Hartree by FFT, the
  two-point KEDF convolutions, the local pseudopotential, XC. This is ADR-0006's finding
  for reduced-order Kohn-Sham, and the model-reduction literature says it in general
  (Chaturantabut and Sorensen, SIAM J. Sci. Comput. 32, 2737 (2010): the nonlinear term
  keeps the full cost unless it is hyper-reduced, e.g. by DEIM). M-OFDFT (Zhang et al.,
  Nat. Comput. Sci. 4, 210 (2024)) is fast only because Hartree is closed-form in its
  Gaussian basis and the grid terms are *learned away*. So C1 is admitted only as an
  instrument to separate basis bias from kinetic-functional error, and must not claim a
  speed-up without hyper-reduction.
- **"Goal-oriented DFT" as stated cannot work (C2).** A self-consistent density needs every
  occupied state, so "solve only the states the target needs" applies only after the SCF,
  where `solver: bands` already does it. The workable form is target-oriented *error
  estimation and adaptivity*, and that is what R1 is.
- **Adaptive resolution is not a plane-wave idea (C4).** A plane-wave basis has one global
  resolution; varying it in space means finite elements, wavelets or local orbitals.

## Literature check (2026-09-30)

- **C2 -- goal-oriented error control.** Established: a posteriori energy and force error
  estimates for plane-wave Kohn-Sham, insulators, fixed k-mesh (Cances, Dusson, Kemlin,
  Levitt, SIAM J. Sci. Comput. 44, B1312 (2022), arXiv:2111.01470; automatic-
  differentiation form: Schmitz, Ploumhans, Herbst, npj Comput. Mater. (2025),
  arXiv:2509.07785); adaptive Ecut for accuracy (Liu, Chen, Dusson, Fang, Gao, Multiscale
  Model. Simul. 20 (2022), arXiv:2012.14806); dual-weighted-residual theory (Becker and
  Rannacher, Acta Numer. 10, 1 (2001)). Open: a controller from a target tolerance to
  Ecut / SCF tolerance / k-mesh; the k-point and smearing error; stress; PAW; energy
  *rankings* (nothing beyond the empirical Carbogno et al., arXiv:2008.10402). Novelty:
  re-implementing the estimates L (85%); the closed loop M (65%, and a natural next step
  for the DFTK group -- a scoop risk); stress M (65%); PAW M (50%); rank preservation
  M-H (60%). The concept "goal-oriented DFT" itself is not new.
- **C1 -- reduced density basis.** Every ingredient has precedent (85%): M-OFDFT
  (molecules, learned functional); Ke et al., Phys. Rev. Lett. 111, 066402 (2013) and
  Phys. Rev. B 89, 155112 (2014) (periodic, non-ML, atom-centred basis, aimed at
  accuracy); Chan, Cohen, Handy, J. Chem. Phys. 114, 631 (2001). Unreported (65%): the
  exact combination of periodic plane waves, WT/WGC with the nonlocal pseudopotential
  functional, and a per-element deformation basis from Kohn-Sham POD. L-M (U).
- **C5 -- embedding.** (a) KS-in-OF in plane waves is L (85%): Trail and Bird, Phys. Rev.
  B 62, 16402 (2000); Genova, Ceresoli, Pavanello (eQE, 2014/2017); Carter's DFET line.
  (b) No paper found that caches or indexes embedding potentials by local environment:
  M (60%). Its real competitor is active-learning MLIPs (Podryabinkin et al.,
  arXiv:1802.07605), which already serve the amortisation use case. Known failure modes:
  large subsystem overlap and partitions across covalent bonds (Genova 2014; Yu and
  Carter, PNAS 2017; Mi and Pavanello, J. Phys. Chem. Lett. 2020). No crossover size is
  reported; an inferred one of several hundred atoms puts it above amaterasu's range.
- **C3 -- frozen subspace.** No direct prior art found, M (65%). Closest: complementary-
  subspace methods (Banerjee, Lin, Suryanarayana, Yang, Pask, JCTC 2018), partial
  Rayleigh-Ritz (Michaud-Rioux, Zhang, Guo, J. Comput. Phys. 2016), an energy window with
  a perturbative correction for band structures only (Allen et al., arXiv:2503.07538).
  Physics caveat: semicore orbitals move with their ion, so a fixed subspace errs at first
  order in the displacement and makes forces non-variational. An ion-following freeze
  (each atom's frozen orbitals translated with its displacement) might remove the first-
  order error; that is unsourced and is recorded as C3's re-evaluation trigger.

What would settle the (U) marks: the full texts of CDKL22 and Ke 2014, and the forward
citations of Ke 2014, M-OFDFT and CDKL22.

## Decision

**Practical track, in priority order.** P1 the `context=` contract of ADR-0013 plus per-
stream extrapolation (B1) -- the foundation every other item uses, and it works for the
ASE calculator too; P2 Harris-Foulkes (B3); P3 LDOS preconditioning (B5), before any slab
or surface work; P4 XL-BOMD (B2) inside the native MD integrator; P5 preconditioned /
low-rank-Hessian relaxation (B6) inside the native relaxation and transition-state
solvers. B4 and B7 held, B8 conditional on hybrids.

**Research track, in priority order.** R1 rank-preserving goal-oriented error control
(C2+C6), which depends only on the Kohn-Sham forces and stress that already exist and on
P2, and not on Phase 10; R2 the KS-minus-OF correction library (C5b), after orbital-free
M3 and C5a; R3 the reduced density basis (C1), after M3, as an error-separation
instrument; R4 the frozen subspace (C3), held. The Phase 10 order of ADR-0001..0011 is
unchanged.

**R1 is validated on our own structure families**, because no full-precision search
history exists to test against. The reference is a high-precision amaterasu run on the
same cell with the same pseudopotential, so the cells may be physically too small: what
is measured is the stability of a ranking under discretisation, not physical truth.
Three families, each run with norm-conserving and PAW datasets:

1. Si polymorphs -- diamond, lonsdaleite, BC8, beta-Sn, simple cubic, fcc (an insulator,
   with the close diamond-lonsdaleite pair, about 10-20 meV/atom in DFT (U));
2. Si-Ge orderings -- every symmetry-distinct configuration of an 8-atom cell, enumerated
   with spglib by our own code (no enumlib), meV-spaced; whether the PseudoDojo standard
   Ge file carries 3d valence electrons is unchecked (U);
3. Al fcc / hcp / dhcp -- a metal, sensitive to smearing and k-mesh, about 30 meV/atom
   between fcc and hcp (U); this family exercises exactly the k-point error CDKL22 leaves
   open.

The (U) energy differences only justify calling the pairs close; they are not gate
values, and are to be sourced when R1 is implemented. Metrics: Kendall tau, top-k recall,
and the largest energy difference of any inverted pair.

**Placement.** P1-P3 join Phase 9; C5a, R2 and R3 join Phase 10 as optional post-M3 items;
P4, P5 and the solver port are the new Phase 11 (ADR-0013); R1 is the new Phase 12, whose
first milestone is adding the rank metrics to `amaterasu convergence`, then CDKL-type
estimates, then the closed loop. Phase numbers are labels, not an execution order.

## Re-evaluation triggers

- C7, C9a, C10, C11: a user decision to pair amaterasu with an MLIP, which is what
  ADR-0001 said makes configuration compression and learned surrogates pay.
- C12, C13 and B8: hybrid functionals being added.
- C4: Phase 8's mixed basis failing its gate of reaching the same accuracy at a lower
  cutoff for localised states.
- C8: amaterasu running systems of thousands of atoms.
- C3: a demonstration that an ion-following freeze removes the first-order force error.
- C9b: B1 falling short on a workload where the first ionic step dominates.
- B4: the Phase 9 profile showing subspace diagonalisation or orthogonalisation, not the
  FFT, as the leading cost.
