---
status: accepted (amended by 0015, 0016)
date: 2026-09-15
---

# Phase 10 is orbital-free DFT with non-ML kinetic functionals, plus a reduced-order Kohn-Sham track; nothing that predicts properties with a neural network

> **Amended by [ADR-0015](0015-general-purpose-scope-and-standard-feature-roadmap.md) and
> [ADR-0016](0016-identity-against-dftpy-error-estimated-multi-level.md).** The scope below
> stands, and the full kinetic-functional ladder is still implemented. Two things change:
> the minimal set (TF, vW, LKT, WT and one density-dependent kernel) goes first, because
> DFTpy already covers the ladder and amaterasu's distinct contribution lies elsewhere
> (ADR-0016); and after Phase 5 the standard-feature and research tracks alternate
> (ADR-0015). The orbital-free engine must also beat VASP above a published atom count N\*
> (ADR-0015).

Two AI-generated surveys (eight documents under `references/`, kept out of the repository)
converged on the same map of ways to speed up DFT without a machine-learned interatomic
potential. Most of their recommendations -- DeepH-style Hamiltonian learning, equivariant
density-matrix prediction, learned kinetic functionals such as M-OFDFT -- put a neural
network in the physics. We decided against all of those, and Phase 10 is instead:

1. **Orbital-free DFT with non-ML kinetic energy density functionals (KEDFs)** -- the core.
   The ladder TF -> von Weizsacker -> TF+lambda*vW -> GGA (LKT) -> nonlocal (Wang-Teter,
   then Wang-Govind-Carter) is implemented first, faithfully and with validation, *as the
   instrument*; designing a new functional is the research goal that instrument enables,
   and is not on the critical path.
2. **Reduced-order Kohn-Sham DFT** by POD-Galerkin projection of Kohn-Sham orbitals from
   snapshots -- second track. It stays Kohn-Sham (no kinetic functional), is a linear SVD
   with no learning, and is scoped to fixed-cell, fixed-composition parameter sweeps
   because a fixed reduced basis does not follow arbitrary atomic motion.
3. **Tensor-train (QTT) compression** is limited to a *rank survey*: measure the QTT rank
   of the density, orbitals and nonlocal kernels on the plane-wave grid. Implementing QTT
   convolutions for the nonlocal KEDF is conditional on that survey showing low rank; at
   the grid sizes this code runs, FFT is already O(N log N) and the QTT payoff is unproven.

Explicitly out of scope, so that nobody re-proposes them in six months:

- **Density-matrix purification / low-rank density-matrix DFT ("Route C")**: needs a
  localised basis; this is a plane-wave code.
- **Latent Hamiltonian / DFTB-style tight binding ("Route D")**: a different theory with
  its own parameter tables, already served by DFTB+; it would reuse nothing here.
- **Compression of the nuclear configuration space** (descriptor PCA -> latent-space
  relaxation or MD): only pays off paired with an MLIP, and the user's interest is in
  compressing the *electronic* representation, not the atomic one.

The constraint behind all of this is the user's: methods whose answer comes out of a
trained black box are not interesting to build. Dimension *reduction* (SVD/POD,
autoencoders, tensor networks) is acceptable; dimension reduction that *replaces the
physics with a regression* is not.
