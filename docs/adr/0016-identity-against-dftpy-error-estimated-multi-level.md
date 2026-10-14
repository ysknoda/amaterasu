---
status: accepted
date: 2026-10-15
amends: ADR-0001 (kinetic-functional priorities), ADR-0012 (research track)
---

# Against DFTpy, amaterasu's identity is error-estimated multi-level electronic structure on top of nonlocal-pseudopotential orbital-free DFT; two research items are added (R5 partial-orbital DFT as a measurement study, R6 the Zhou-Wang step reused as an error estimator) and "certified" is reserved for rigorous bounds

DFTpy (Pavanello group; Shao, Jiang, Mi, Genova and Pavanello, arXiv:2002.02985;
latest release 2.2.0, May 2026) is a
plane-wave orbital-free code in Python -- the same concept as amaterasu's Phase 10. Doing
what it already does would not be interesting. This record says where the overlap is,
what amaterasu does that DFTpy does not, and which new ideas were adopted after a
literature check.

## What overlaps and what does not

The overlap is the kinetic functional ladder (ADR-0001's M1-M3): DFTpy has TF, vW, WT,
the MGP family and density-dependent kernels (LWT/LMGP by spline interpolation in k_F),
LKT, MPI, spin polarisation and time-dependent orbital-free DFT, and has run a
million-atom Al cell. Speed is therefore not a differentiator (ADR-0015 still requires
amaterasu to beat VASP, which is a different claim).

What DFTpy does not have, as far as the paper, the manual and the release notes show:

- **Nonlocal pseudopotentials in orbital-free DFT.** DFTpy uses local pseudopotentials
  (BLPS/OEPP). amaterasu's ADR-0008 uses PseudoDojo files with their projectors through
  the nonlocal pseudopotential functional (Xu et al., Nat. Commun. 13, 1385 (2022)).
- **Its own Kohn-Sham engine on the identical file**, norm-conserving and PAW, with
  finite-difference-checked forces and stress. DFTpy's Al Kohn-Sham-versus-orbital-free
  comparison does not say whether the Kohn-Sham side used the same local
  pseudopotential, which is exactly the confound ADR-0003 was written to avoid.
- **Error estimation** of any kind.
- GPU support (amaterasu has none either; not a claim).

## Decision

**Identity (B2): error-estimated multi-level electronic structure.** The theory level
(orbital-free -> partial-orbital -> Kohn-Sham -> PAW) and the discretisation are chosen per
structure (screening) or per region (embedding) from a posteriori estimates of the error
in a goal quantity -- an energy ranking, forces. R1 (ADR-0012) supplies the discretisation
part, R6 below the orbital-free part, R5 the intermediate level. The nonlocal
pseudopotential functional (ADR-0008) is the foundation, because it is what lets
orbital-free and Kohn-Sham consume the same file.
Novelty of the framework: medium-high, about 70 %, from a literature search.

**Naming.** No rigorous bound on the orbital-free-versus-Kohn-Sham error is known, so a
choice between those levels is a *calibrated estimate*, not a guarantee. The umbrella is
called **"error-estimated multi-level"**; "certified" is used only for parts with a
rigorous bound, such as discretisation errors where one exists.

**Kinetic functionals.** ADR-0001's ladder is implemented in full, but the minimal set
goes first: TF, vW, LKT, WT and one density-dependent kernel (WGC is the expected choice;
it is settled at M3). DFTpy becomes an **opt-in test oracle**: installed in an isolated
virtual environment, never a dependency, compared through `pytest -m dftpy` on the BLPS
Al and Si cells, like the akitsushima cross-check of ADR-0013. Permission to install it is
asked when M1 starts.

**R6 -- the Zhou-Wang step reused as an error estimator (formerly "O1").** If the kinetic
functional were exact, the orbital-free density would be the ground-state density of
non-interacting electrons in the orbital-free effective potential v_s. One Kohn-Sham
diagonalisation in v_s gives that density and its kinetic energy. The construction is
**not new**: Zhou and Wang, J. Chem. Phys. 124, 081107 (2006) published it as an energy
correction, and Thapa et al., J. Chem. Phys. 158, 214112 (2023) use it as an energy-level
diagnostic. What is new here is the use: a spatial map of the inconsistency, and a
benchmark of the indicator. Known weaknesses and the answers to them:

- the density residual is smoothed by the response function chi_s, and a small residual
  can coexist with a large energy error -> a Harris-functional energy is computed
  alongside, sharing ADR-0012's P2 implementation;
- error from the nonlocal pseudopotential functional contaminates the residual -> a control run with an identical external
  potential separates it.

Gate: the indicator correlates with the actual |E_OF - E_KS| over Li, Mg, Al, Si and Be,
and is large for Be, the known failure. Placed after M2. Novelty of the map plus
benchmarked indicator: medium, about 60 %.

**R5 -- partial-orbital DFT (formerly "B1", renamed to avoid ADR-0012's row B1).** Solve
the lowest M orbitals explicitly and treat the remaining density rho - sum |phi_i|^2 with a
Pauli functional; M = 0 is orbital-free, M = N/2 is Kohn-Sham. Finite-temperature
relatives exist -- Hollebon and Sjostrom, PRB 105, 235114 (2022); White and Collins, PRL
125, 055002 (2020); extended FPMD; Sadigh, Åberg and Pask, PRE 108, 045204 (2023) -- but a
zero-temperature scheme for the occupied remainder was not found (about 75 %). The central
difficulty is that the remainder's kinetic energy depends on the explicit orbitals through
orthogonality, the same non-additive kinetic energy that troubles subsystem DFT, where
local approximations are off by 30-80 % for overlapping densities. Therefore **the first
deliverable is a measurement study, not a functional**: for Al (metal) and Si (covalent),
measure the exact T_rest(M) and the non-additive Pauli term with amaterasu's own
Kohn-Sham, and map where and by how much local approximations fail. Whether to build a
functional is decided from that. Novelty: medium, about 65 % (UNCERTAIN). Scoop risk: an
ICMRE-2026 abstract by Ma, Shao et al. on "low energies by Kohn-Sham, high energies by
orbital-free", aimed at warm dense matter.

**O2 -- transferring a measured kernel (ADR-0010) to large systems: held.** No prior art
was found (about 80 %), but the kernel measurements come from the DFTpy group itself, and
the risk of being scooped within twelve months is about 30 %. ADR-0010's measurement is
still built; transfer is not made a research claim.

**O3 -- choosing the Kohn-Sham region from the R6 error map: merged** into ADR-0012's C5a
and R2, after R6. It needs a full Kohn-Sham diagonalisation and its interface errors
dominate, so it does not stand alone. Novelty: medium-high, about 70 %.

**Research order** (alternating with ADR-0015's standard groups): M0 -> M1 minimal (TF,
vW, LKT) -> `solver: response` -> M2 (WT) -> one density-dependent kernel -> orbital-free
numba/MPI and the N\* measurement of ADR-0015 -> nonlocal pseudopotential functional -> R6 -> R5 measurement study -> the
rest of the ladder -> C5a/R2 including O3 -> Phase 12 (R1). Held: Phase 8, R3, R4, O2.

## How the novelty figures were obtained

Three read-only literature searches (2026-10-15), web searches read largely through
abstracts plus full text where available. Percentages are the searcher's confidence that
no prior work does the same thing; all are below 90 % and are UNCERTAIN. Two citations
from the first round were corrected: arXiv:2409.12625 is Electron. Struct. 7, 015006
(2025) -- PRB 108, 235168 (2023) is a different paper; arXiv:1908.05721 (density-corrected
DFT analysis) is Vuckovic, Song, Kozlowski, Sim and Burke, J. Chem. Theory Comput. 15,
6636 (2019).
