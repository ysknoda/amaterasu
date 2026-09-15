---
status: accepted
date: 2026-09-15
---

# Measure the kinetic energy kernel with this code's own Kohn-Sham engine instead of fitting around it; expose it as `solver: response`

The kinetic energy kernel is the second functional derivative of the kinetic energy with
respect to the density, and it is what every nonlocal kinetic energy density functional
is built from: Wang-Teter fixes it to the Lindhard function of the uniform gas, Huang-
Carter to a uniform-gas-with-gap model. Moldabekov, Shao, Pavanello, Vorberger and
Dornheim (arXiv:2409.12625) showed it can simply be measured: apply a static harmonic
perturbation to a Kohn-Sham calculation, read off the density response, and invert. Their
result for fcc Al and diamond Si is that the model kernels are excellent out to
`q = 2*pi/(2*r_cut)` and wrong beyond it, and that the error is a consequence of using
nonlocal pseudopotentials -- the region the projectors cover.

That is the defect ADR-0009's `A` and `q` are patching with two scalars, and it is the
reason ADR-0008's element ceiling exists. It is also, unusually for this problem, cheap
to measure: everything needed is already in the Kohn-Sham engine except one extra term in
the external potential.

## Decision

- Add `solver: response`, alongside `scf`, `bands`, `relax` and the rest, taking
  `response: {perturbation: harmonic, qpoints: [...], amplitude: ...}`. It writes the
  density response, the Kohn-Sham response and the kernel into a `response/` group in the
  HDF5 output, and `amaterasu plot kernel` draws them.
- Verify the linear-response assumption rather than assuming it: run several amplitudes
  and show the response is linear in the one chosen, as the source paper does before
  using its own method.
- Place it directly after M1, once the semilocal functionals exist, so that Thomas-Fermi,
  von Weizsacker, TF-lambda-vW and LKT can be compared *as kernels* before Wang-Teter and
  WGC are built on top of them.

## Not the Sternheimer route

Phase 6 (DFPT) will solve the Sternheimer equation for electric-field, phonon and strain
perturbations, and putting this there would look like reuse. It is the wrong mechanism:
what is wanted is the self-consistent response to a *finite* harmonic perturbation, which
is the source paper's "direct harmonic perturbation approach". Adding one term to the
existing external potential is enough, and it removes a dependency on Phase 6 that would
otherwise push this past M3.

## XWM is deliberately not implemented

Xu et al. run their scheme with both Wang-Teter and their own XWM functional. XWM is not
added to the ladder: the layer-3 verification of ADR-0009 reaches all seven of their
elements through Wang-Teter alone (Be, Zn and Cd have no XWM row anyway), and the fact
that XWM would demonstrate -- that `A` and `q` depend on the kinetic functional -- is
already visible in their own table. Measuring the kernel answers the same question more
directly and for every functional at once.
