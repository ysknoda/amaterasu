---
status: accepted
date: 2026-09-15
---

# Orbital-free DFT here is an sp-element, nearly-free-electron method; that ceiling is physical and no pseudopotential generator lifts it

The BLPS Library covers nine elements (Li, Mg, Al, Si, P, Ga, As, In, Sb), so the obvious
worry was that orbital-free DFT would be limited by data. It is not; it is limited by the
theory, and the two papers that define the method say so themselves (Zhou, Wang, Carter,
Phys. Rev. B 69, 125109 (2004), sections I-II):

- A local pseudopotential acts with one potential on every angular-momentum channel, so
  "there is no way to have such a potential reproduce all the valence eigenvalues". That
  is the whole reason nonlocal pseudopotentials exist, and it is why a local one is a poor
  description of any element whose valence channels scatter very differently -- d-metals
  above all.
- Linear-response kinetic functionals (Wang-Teter, WGC) reach meV accuracy "for sp-bonded
  nearly free-electron-like metals" and "do not achieve meV accuracy for systems with
  localized electron densities such as in molecules and covalent or ionic solids", for
  which "no simple yet accurate KEDF's are available".

So transition metals, and oxygen/nitrogen/carbon compounds, are out of reach for reasons
that sit in the kinetic functional, before the pseudopotential is even chosen. A perfect
local-pseudopotential generator would extend the method to more *sp* elements (Na, K, Ca,
Ge, Sn, Pb, ...) and no further. The nine elements in the library are the elements the
method's authors found it works for.

Decisions:

1. **Phase 10 does not claim generality.** Its deliverables are an orbital-free engine
   that is compared against Kohn-Sham on identical pseudopotentials (ADR-0003) within the
   sp domain, and a set of kinetic functionals that serves as the instrument for
   developing new ones. Documentation and error messages must say "orbital-free theory is
   for sp-bonded nearly-free-electron systems", not "not yet supported".
2. **A BLPS generator is an optional milestone M5, after M3, behind a go/no-go gate.** The
   2004 procedure is: target Kohn-Sham densities of several bulk phases with a good
   nonlocal pseudopotential; Wang-Parr inversion of the Kohn-Sham equations with the
   eigenvalue-shift stabilisation; unscreening; division by the structure factor and
   spherical averaging over phases; spline smoothing with the sub-Bragg small-g values
   tuned against multi-phase equations of state; and a real-space Coulomb-tail
   enforcement that needs an atomic local pseudopotential as its base. amaterasu already
   has the hardest prerequisite -- a plane-wave Kohn-Sham solver that accepts an arbitrary
   grid potential -- but the inversion, the small-g tuning loop and the atomic base
   (which would come from an atom-in-a-box calculation, since this code deliberately has
   no radial atomic solver) are each a small project; together they exceed M1-M3. The
   gate is: reproduce the library's aluminium BLPS -- its v(g) curve and the Al fcc/bcc/hcp
   equations of state -- from our own Kohn-Sham densities.
3. **A documented second-class route for elements outside the nine**: a pseudopotential
   generated with an external tool (ONCVPSP, APE) with the least-bad channel chosen as
   the local one and the other channels dropped at generation time. Transferability is
   worse than BLPS, as the 2004 paper states; it is written down so the option is known,
   and no code is built for it.
4. **The library's Al and Si `.psp` files may be bundled as test fixtures unchanged**, with
   attribution, under the library's CC BY-ND 4.0 -- the reader consumes the original
   file. **No converted or modified form may be distributed**, so no `.recpot`, no
   re-gridded or re-formatted copy, ever goes into the repository.
