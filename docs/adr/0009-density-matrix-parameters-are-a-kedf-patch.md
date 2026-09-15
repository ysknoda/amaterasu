---
status: accepted
date: 2026-09-15
---

# The density-matrix parameters `A` and `q` are a patch on the kinetic functional, not a property of the pseudopotential: ship an uncalibrated default, calibrate against the isolated atom, and use the published values only to verify our implementation

The modified-Gaussian density matrix that makes ADR-0008 work carries two numbers:

    gamma(r,r') = rho_q(r,r') * exp(-s^2 / 2*beta) * [1 + A * (s^2 / 2*beta)^2]

with `s = |r-r'|`, `beta(r,r') = 3*rho / (2*t_s)` the local temperature, and
`rho_q = [(rho(r)^q + rho(r')^q)/2]^(1/q)` a q-mean. `A` traces back to Lee and Parr
(Phys. Rev. A 35, 2377 (1987)) as the O(s^4) coefficient of a Taylor expansion, but Xu et
al. do not use the derived value: their text calls it "an adjustable parameter" and their
Methods say both numbers are "carefully tuned to yield the bulk properties, which agree
with the KS-DFT (NLPPs) predictions". The values are in their Supplementary Information,
Supplementary Table 2 -- **all 10 rows, 7 elements, reordered by atomic number**:

| Z | Element | KED | `A` | `q` |
|---|---|---|---|---|
| 3 | Li | WT | -0.851 | 0.999 |
| 3 | Li | XWM | -0.908 | 1.155 |
| 4 | Be | WT | -1.683 | -0.005 |
| 12 | Mg | WT | 5.383 | -2.140 |
| 12 | Mg | XWM | 4.500 | -1.512 |
| 19 | K | WT | 0.310 | 0.132 |
| 19 | K | XWM | 0.139 | -0.263 |
| 30 | Zn | WT | 1.027 | -0.185 |
| 48 | Cd | WT | -0.655 | -0.478 |
| 55 | Cs | WT/XWM | 0.590 | 0.095 |

Two irregularities in the source table, recorded rather than smoothed over: Cs has a
single row shared by both kinetic functionals where every other element has one row per
functional, and Be, Zn and Cd have no XWM row at all.

Three things follow, and they decide the policy:

1. **There is no derived value to fall back on.** The paper does not use Lee and Parr's
   coefficient, and gives no derivation of its own.
2. **There is no pattern across the periodic table.** Going down the alkali metals,
   `A` runs -0.851 -> +0.310 -> +0.590, changing sign; `q` runs 0.999 -> 0.132 -> 0.095.
   The two alkaline earths are `A = -1.683` (Be) and `A = +5.383` (Mg). Same group, same
   valence configuration; nothing to interpolate.
3. **The values depend on the kinetic functional, not just the element.** Li is -0.851
   with Wang-Teter and -0.908 with XWM; Mg is 5.383 and 4.500. A property of the
   pseudopotential could not do that. These two numbers are absorbing kinetic-functional
   error: `A = 5.383` does not describe magnesium's density matrix, it describes how
   badly Wang-Teter does for magnesium.

Point 3 also explains where the error is. `A` and `q` reshape the kernel at short range,
which is large `q` in reciprocal space -- exactly the region where Moldabekov et al.
measure the kinetic kernel departing from the built-in von Weizsacker limit once nonlocal
pseudopotentials are used (ADR-0008). Fitting them is a two-scalar repair of a defect
that ADR-0010 proposes to measure directly.

## Decision -- three layers, kept separate

- **Layer 1, the default: `A = 0`, `q = 1`.** The plain Lee-Parr Gaussian with an
  arithmetic mean density. No fitting, applicable to all 85 elements, and it will not
  reproduce Xu et al. Whatever it misses by is recorded as a measurement. This is what
  ships, and what an uncalibrated element runs on.
- **Layer 2, calibration (the replacement for the retired M5): fit `A` and `q` per
  element against this code's own Kohn-Sham calculation of the *isolated atom*, not of
  the bulk.** The bulk fit that Xu et al. use makes the bulk gate self-referential -- a
  quantity tuned to reproduce `V0` and `B0` will reproduce `V0` and `B0`. An
  isolated-atom target keeps the bulk gate independent, follows the pseudopotential
  tradition of generating on the atom and validating on the solid, and costs seconds;
  amaterasu already runs isolated atoms (Phase 4 matched a Si 3p-3s splitting to the
  dataset's own all-electron eigenvalues within 2.8 meV). Whether an atom-fitted pair
  transfers to the solid is not known -- if it does, that is a small real contribution;
  if it does not, the reason why the bulk is needed is itself worth knowing.
- **Layer 3, implementation verification: take Xu et al.'s published `A` and `q` as fixed
  inputs and try to reproduce their numbers.** This is not fitting. It tests the double
  integral, `chi_lm`, and the `D_ij` coupling against an independent implementation.

Calibrated pairs live in their own manifest, **keyed by the pseudopotential file's
SHA256**, not in the download registry. Keying on the hash means a calibration can never
be silently applied to a file that has since changed -- the same class of quiet-wrong
failure as most of the real bugs found in Phases 2 to 4. The manifest records the fit
target, the kinetic functional, and when it was produced; the log states on every run
which of the two states is in force.
