---
status: accepted
date: 2026-09-15
---

# Neural networks enter as an optional kinetic energy density functional, not as an exchange-correlation functional; the analytic ladder stays the default and the gate table decides which wins

The original Phase 10 brief excluded machine learning that predicts a physical property
directly, on the grounds that a black box is not an interesting answer (ADR-0001).
ADR-0008 and ADR-0009 then established that the kinetic energy density functional is the
*only* remaining source of error in the orbital-free engine: the pseudopotential ceiling
is gone, and the two density-matrix parameters that remain are themselves a patch on the
kinetic functional. Machine-learned kinetic functionals are therefore no longer a
tangential idea; they are a direct competitor to the work this phase is about, and they
have to be treated as one.

## They are real, and they are two different things

Exchange-correlation and kinetic functionals are in very different states:

- **Exchange-correlation.** Skala (Microsoft, arXiv:2506.14665, weights published) reaches
  2.8 kcal/mol on GMTKN55 and 1.06 kcal/mol on W4-17 at semi-local cost -- better than
  hybrids. DM21 (DeepMind) needed a second-order perturbation regulariser holding its
  orbitals near B3LYP to be self-consistently stable, and still does not give physical
  densities for stretched polar bonds. All of the demonstrated wins are **main-group
  molecular thermochemistry**; transfer of molecule-trained descriptors to periodic
  solids is described in the literature as a regime change that is rarely tested.
- **Kinetic.** CPN5 (arXiv:2408.15062, a multi-channel nonlocal network in ABACUS) cuts
  the charge-density error on semiconductors from 8.82% to 3.23% in-sample and 8.96% to
  7.59% out-of-sample, trained on 196,830 grid points from ten structures. Imoto, Imada
  and Oshiyama (Phys. Rev. Research 3, 033198 (2021)) learn the functional *derivative*
  on grid points across 24 atoms, molecules, metals, semiconductors and an ionic solid.
  A third line (arXiv:2502.05411, Mach. Learn. Sci. Technol. 2025) uses a hybrid
  Gaussian-process/network model only to *understand* the functional dependence and
  ships a **closed-form analytic functional**, covering Li, Na, Mg, Al, Si, P, Ga, As,
  In, Sn and Sb -- eleven sp elements, the same ceiling ADR-0007 records.

## Decision

- **No neural exchange-correlation functional.** amaterasu is a periodic-crystal code
  whose exchange-correlation path already agrees with libxc to 3e-15 and whose PBEsol
  results pass the Delta gate on five elements. The published gains are in a domain this
  code does not target, while the costs land squarely inside it: a tensor-library
  dependency, a network evaluated on the FFT grid every iteration in Python, `v_xc` by
  automatic differentiation, and forces and stresses needing both the density derivative
  and the position derivative of the network -- on top of the self-consistency stability
  problem DM21 had to regularise away. Recorded, not implemented.
- **Two machine-learning routes into the kinetic functional, both kept.** The user asked
  that both be available so that the choice stays with whoever runs the code:
  1. **Offline discovery, analytic product.** Measure `K_s(q)` with `solver: response`
     (ADR-0010) for real pseudopotentials, find where it leaves the Lindhard form, and
     fit an analytic correction. No network at run time and no new dependency; the
     result is an expression that can be differentiated by hand for forces and stress
     and checked against finite differences like everything else here.
  2. **A run-time neural kinetic functional**, selected as one more entry in `kedf:`
     alongside `tf`, `vw`, `tflambdavw`, `lkt`, `wt`, `wgc`, and built on PyTorch (see
     below). Its real costs, recorded here so they are not discovered later: the torch
     dependency, training-set generation, and **the stress**.
- **The analytic ladder remains the default.** A network is opt-in.

### What a neural kinetic functional does *not* cost: forces

An earlier draft of this record claimed that forces would have to be implemented for a
neural functional because the CPN5 paper does not report them. That was wrong, and the
correction matters because it makes route 2 cheaper than it looked. With

    E[rho] = T_s[rho] + E_H + E_xc + E_ext[rho, R] + E_II(R)

the density is variational, so at the minimum the implicit term vanishes and `dE/dR`
keeps only the explicit `R` dependence. `T_s[rho]` has none. **No kinetic functional,
neural or analytic, contributes an explicit force term** -- which is what this project's
own plan already said for the semilocal functionals at M1, and it does not stop being
true because the functional is a network.

Two real costs sit next to that, and they are the ones to watch:

- **Stress does have an explicit term.** Straining the cell changes the G vectors and the
  reference density that the nonlocal kernels depend on. This is new work of the same
  shape as the WGC kernel stress at M3.
- **Force quality tracks how tightly the minimisation converges**, because the force is
  only free once the density really is at the variational minimum. A functional whose
  derivative is noisy gives noisy forces with no explicit term to blame. This is the
  likelier reason CPN5 reports no forces, and it is a gate concern, not an implementation
  one.

## These are not competing axes, and the configuration says so

The nonlocal pseudopotential functional (ADR-0008) and the kinetic energy density
functional are orthogonal: every orbital-free run needs one of each, and a neural kinetic
functional runs *with* the nonlocal pseudopotential functional, not instead of it. The
config keeps them as separate keys (`electrons.nonlocal_pseudo` and `electrons.kedf`),
which makes the full grid of combinations available rather than forcing a choice between
two things that were never alternatives. The real contest is between kinetic functionals:
neural against Wang-Teter, WGC and whatever the measured kernel suggests.

## What settles the contest

The gate table of ADR-0008, unchanged, applied to every kinetic functional on the same
footing: equilibrium volume within 1%, bulk modulus within 10%, correct relative-energy
ordering between phases, and finite-difference-verified forces and stresses. Our own
measurements decide, not published claims.

This is a real test and not a formality, because the two things a network is currently
weakest at are the two things the table weighs most. CPN5's authors state plainly that
their functionals "are unable to reproduce the energy ordering of different phases of
Si", attributing it to training data that does not span volumes and phases and to
spherically symmetric kernels that cannot carry anisotropy; and they do not report forces
at all. A functional that predicts better densities but the wrong phase ordering is not usable
for the relaxations and molecular dynamics this code exists to run, however good its
density looks.

**Re-evaluation trigger**, recorded so that the decision is not frozen by inattention: a
published neural kinetic functional that (1) reports forces, (2) reproduces phase energy
ordering, and (3) transfers to elements outside its training set, reopens the priority
of route 2 against the analytic ladder.

## How route 2 is built (PyTorch)

**Where torch lives.** A `mlkedf = ["torch>=2.2"]` optional extra, included in `all`, on
the same footing as `jit = ["numba"]` and `mpi = ["mpi4py"]`: a plain install of
amaterasu does not pull it in. The conversion boundary is a single adapter at the KEDF
interface; nothing else in the code leaves numpy. A **numpy reference implementation of
the forward pass** is kept beside the torch one and tested against it, exactly as Phase 9
requires of every numba kernel -- so the test suite runs without torch, and a change in
torch behaviour between versions shows up immediately instead of silently.

**Model.** CPN5's shape is the starting point, being the published state of the art for
solids with its descriptors written down: a semilocal descriptor plus three nonlocal ones
per scaled channel (16 components for five channels), the channel convolutions done with
`torch.fft` inside the same graph, a small constrained network, and `t_s(r)` out.
`dT/drho` comes from autograd -- and is checked against a finite difference of `T`
regardless, because in this project a derivative is not trusted for being automatic any
more than for being textbook.

**Training data comes from our own Kohn-Sham runs.** The gate elements of ADR-0008 (Li,
Be, Mg, Al, Si) across several phases and several volumes -- reusing the plus/minus 6%
seven-point set the Delta protocol already uses. This directly targets CPN5's own
diagnosis of its failure, that its training data spanned neither volumes nor phases.
Two targets are stored per grid point, not one: the kinetic energy density, and the
**functional derivative** `dT_s/drho = mu - v_s(r)`, which is a by-product of a Kohn-Sham
run in our own code and is the quantity Imoto, Imada and Oshiyama argue matters because
it is what appears in the Euler equation. Published weights are not ported: matching
another code's descriptors, grids and pseudopotentials exactly is the kind of work that
proved more expensive than doing it ourselves in Phase 4.

**The first experiment holds beryllium out.** Train on Li, Mg, Al and Si; test on Be,
where Wang-Teter's failure is already measured and recorded as a negative control. The
question it answers is the only one worth asking of a learned functional here: did it
learn something Wang-Teter does not know, or is it interpolating between its training
elements? A hit is real extrapolation; a miss reproduces CPN5's own conclusion -- better
densities, no more physics -- as our own measurement rather than someone else's.

**Order.** After M3, so that the WGC kernel and its stress are available as the template,
and after the kernel measurement of ADR-0010, which says which descriptors are worth
feeding it.
