---
status: accepted
date: 2026-09-15
supersedes: 0002
amends: [0003, 0007]
---

# Orbital-free DFT can use nonlocal pseudopotentials after all, through a density-matrix functional; PseudoDojo replaces BLPS as the production source and the BLPS generator (M5) is retired

ADR-0002 recorded that orbital-free DFT has no orbitals, so a Kleinman-Bylander projector
-- which acts on orbitals -- cannot be applied, and concluded that the engine needs local
pseudopotentials. The first half is true; the conclusion does not follow. The nonlocal
pseudopotential energy does not need orbitals, only the non-interacting density matrix:

    E_nl = sum_i f_i <psi_i| V_nl |psi_i> = int int V_nl(r',r) gamma_s(r,r') dr dr'

and `gamma_s` can be approximated explicitly as a functional of the density, exactly as
kinetic energy density functionals are. Xu, Ma, Mi, Wang and Ma did this in Nat. Commun.
13, 1385 (2022) with the Lee-Parr modified-Gaussian density matrix, and state the point
in their own abstract: the scheme "defies the belief that nonlocal pseudopotentials are
not applicable to OF-DFT". None of the eight survey documents that motivated Phase 10
mentions it, and neither did ADR-0002.

This matters here more than it would in most codes, because amaterasu already reads psp8
and UPF with their projectors and already builds `chi_lm` from them for its Kohn-Sham
engine (`amaterasu/pseudo/base.py`, `kb.py`). The missing piece is the double integral,
not the data.

## Decision

- **Production source becomes PseudoDojo norm-conserving psp8/UPF**, the same files the
  Kohn-Sham engine uses: 85 elements, CC BY 4.0, reader and projector machinery already
  in place and validated in Phase 2.
- **BLPS is kept, demoted to a reference point.** The Princeton `.psp` files have zero
  projectors, so the nonlocal functional must return `E_nl` identically zero on them.
  That is a machine-precision unit test of the whole construction -- the double integral,
  `chi_lm`, and the `D_ij` coupling -- and it is also what lets BLPS and PseudoDojo be
  compared through one code path.
- **M5, the in-house BLPS generator, is retired.** It was gated and estimated as harder
  than M1+M2+M3 combined. Per-element work does not disappear (see ADR-0009), but it
  shrinks from a Wang-Parr inversion producing a whole radial potential to fitting two
  scalars.
- **Both ways of splitting local from nonlocal are supported and measured.** Xu et al.
  use Troullier-Martins potentials with the p channel *as* the local potential; ONCVPSP
  files carry a separately constructed local potential and projectors in every channel,
  so their nonlocal part is larger and the density-matrix approximation carries more of
  the load. `local_channel: dataset | s | p | d` selects between them. Which one is
  better is not known in advance and is a measurement this code can make.

## 85 elements run; 85 elements are not accurate

The ceiling moves from the pseudopotential to the kinetic functional. It does not go
away, and it is not far off:

- **Beryllium**, an sp element with four electrons, already fails: Xu et al. report the
  simple-cubic relative energy off by more than 400 meV/atom and `B0` for hcp at 91.5 GPa
  against 123.3 from Kohn-Sham PAW. They attribute it (their Fig. 4) to the Wang-Teter
  kinetic energy density being badly underestimated where the density varies quickly near
  the core.
- **Cadmium** needed its pseudopotential rebuilt without d-channel electrons before the
  scheme would run at all.
- **Transition metals generally** reach only qualitative accuracy: arXiv:2511.19892
  (2025) reports mean absolute relative errors of 28-37% in `V0` and 55-60% in `B0`
  across the 3d/4d/5d series.
- The reason is now *measured*, not just argued: Moldabekov, Shao, Pavanello, Vorberger
  and Dornheim (arXiv:2409.12625) extract the kinetic energy kernel from Kohn-Sham runs
  and find that below `q = 2*pi/(2*r_cut)` the uniform-gas kernel is excellent for fcc Al,
  while above it, with nonlocal pseudopotentials, the kernel departs linearly in `q` from
  the von Weizsacker limit that Wang-Teter and Huang-Carter build in by construction.

So the code must not present a number for an arbitrary element as if it were validated.

## Element classes

Two independent facts are reported, never merged into one verdict:

- **Accuracy evidence**, from this project's own measurements only:
  - `validated` -- passed the gate below. A short hand-maintained list, each entry
    pointing at the test that put it there.
  - `expected` -- no d or f valence projectors in the dataset, but not gated. Warn.
  - `unvalidated` -- the dataset carries d or f projectors. Warn prominently.
  The second and third are decided from the pseudopotential file itself, so there is no
  hand-maintained periodic table to drift out of date. Published results by other groups
  are *not* evidence for this field; using someone else's measurement as our guarantee is
  the mistake this field exists to prevent.
- **Calibration state** (ADR-0009), stated on every run: either the uncalibrated default
  or the calibrated pair with its provenance.

## Gate (this replaces the element list in ADR-0003; the structure of that gate stands)

The primary gate is unchanged in kind -- amaterasu's own Kohn-Sham run -- but is now
sharper, because Kohn-Sham and orbital-free consume the *identical file including its
projectors*, so nothing but the functional differs.

| Z | Element, phase | Role | Criterion |
|---|---|---|---|
| -- | Al, Si (BLPS, zero projectors) | degenerate gate | `E_nl == 0` to < 1e-10 Ha |
| 3 | Li bcc/fcc/hcp | verification against Xu et al. with their published `A`, `q` | `V0` 1%, `B0` 10%, `ER` ordering |
| 4 | Be hcp, sc | negative control -- known to fail | report the failure; flag if it is not of the reported order, *including if it is too good* |
| 12 | Mg hcp | largest published `A` (5.383); tests whether the uncalibrated default is hopeless or merely worse | report only |
| 13 | **Al fcc** | **primary gate** | vs own KS, same file: abs(d a0) <= 1%, abs(d B0) <= 10% |
| 14 | Si diamond | measured-failure study (ADR-0003) | report only |

Zn (30), Cd (48) and Cs (55) are optional extras for the Xu et al. verification; they are
not gates, and Zn and Cd carry d electrons. Forces and stresses are finite-difference
checked for every element, as everywhere else in this project.

A gate that requires a *failure* is unusual. It is here because a clean Be result would
mean either that Wang-Teter had been accidentally repaired, or that a bug was masking the
kinetic-functional error. The second is far likelier, and nothing else in the suite would
catch it.

## Consequences for earlier records

- **ADR-0002 is superseded.** Its measured facts about the BLPS file format remain
  correct and useful; its conclusion does not.
- **ADR-0003 is amended**: the gate structure ("our own Kohn-Sham, same pseudopotential")
  stands and gets stronger; the element list and the pseudopotential source are replaced
  by the table above.
- **ADR-0007 is amended, and its conclusion stands.** Orbital-free DFT here is still an
  sp-element method -- but of the two reasons recorded there, only one survives. The
  first ("a local pseudopotential acts with one potential on every angular-momentum
  channel") is now moot, because the potential no longer has to be local. The second
  (linear-response kinetic functionals fail for localized densities) not only survives
  but is reinforced by the kernel measurements above. Documentation and error messages
  keep saying "sp-element method", for the second reason alone.
