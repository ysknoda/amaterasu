# amaterasu

> **Alpha release.** This project is under active development.
> Interfaces, behavior, and output formats may change without notice
> between releases, and it has not yet been broadly tested. Do not
> rely on it for production use.

Plane-wave / mixed-basis density functional theory for three-dimensional periodic crystals.

> **Status: early development (Phase 4 complete, Phase 5 under way).** `solver: scf`, `relax` and `md` all
> run, with norm-conserving *and* PAW datasets. Plane-wave DFT with LDA/PBE/PBEsol, spin
> polarisation, smearing and symmetry-reduced k-points; analytic forces and stress; and an
> [ASE calculator](#ase-calculator) that drives ASE's own optimisers, molecular dynamics
> and (from Phase 5) nudged elastic band. Silicon reproduces the PseudoDojo reference
> equation of state to 0.04 meV/atom and the forces match finite differences to
> 1e-5 eV/Å; the PAW delta factor against all-electron WIEN2k is under 3 meV/atom for
> every element tried -- see [Validation](#validation). PAW reads both PAW-XML (Blöchl)
> and UPF (Kresse-Joubert), so PBEsol + PAW works. The compensation-charge shape
> dependence is measured, bounded and documented; the Gaussian stays the default.

## What it is

A from-scratch Kohn-Sham DFT code in Python, built on numpy/scipy for the numerics,
numba for the hot loops, and ASE, spglib, seekpath and phonopy for the surrounding
infrastructure. amaterasu implements the electronic-structure engine -- plane-wave basis,
pseudopotentials and PAW, exchange-correlation functionals, eigensolvers, SCF, forces,
stress, DFT+U, DFPT, Wannier functions -- and drives the well-tested ASE implementations
for geometry optimisation, molecular dynamics and nudged elastic band.

## Units

amaterasu computes internally in **Hartree atomic units** and converts once, at the output
boundary, so every reported number uses the same units regardless of which internal
quantity produced it:

| quantity | output unit |
|---|---|
| energy | eV |
| length | Å |
| force | eV/Å |
| pressure, stress | GPa |
| time | s |

Input files are a separate convention: each field defaults to the unit its own community
uses (Ry for plane-wave cutoffs, eV for smearing, fs for MD timesteps, GPa for pressure),
and any energy field also accepts an explicit unit, e.g. `ecut: "816 eV"`.

## Installation

```bash
pip install -e ".[test]"
```

Optional extras: `jit` (numba, for the Phase 9 compiled kernels), `mpi` (mpi4py),
`fftw` (pyFFTW), `phonons` (phonopy), `plot` (matplotlib). None are needed for a
plain run; each is imported only where the feature it backs is used, and the pure
numpy/scipy path is taken when it is absent.

## Quick start

A calculation is described by two files: a plain-text **structure** file and a YAML file
of **numerical settings**. Keeping them apart means a parameter sweep varies one while
holding the other fixed.

`Si.str`:

```
units       angstrom

%block lattice_vectors
  0.00000   2.71535   2.71535
  2.71535   0.00000   2.71535
  2.71535   2.71535   0.00000
%endblock lattice_vectors

%block atomic_positions fractional
   Si    0.0000   0.0000   0.0000
   Si    0.2500   0.2500   0.2500
%endblock atomic_positions
```

`calc.yaml`:

```yaml
structure: Si.str
solver:    relax
xc:        {functional: PBE}
pseudo:    {method: nc, family: pseudodojo-nc-sr-pbe-standard-v0.4}
basis:     {ecut: 60}            # Ry
kpoints:   {scheme: monkhorst-pack, grid: [8, 8, 8]}
relax:     {algorithm: fire, fmax: 0.01}
```

```bash
amaterasu info Si.str                 # space group, volume, density, nearest neighbour
amaterasu run  calc.yaml --dry-run    # validate and print every resolved setting
amaterasu run  calc.yaml
amaterasu run  calc.yaml --set kpoints.grid='[4,4,4]' --set basis.ecut=80
amaterasu convert POSCAR -o Si.str

# converge one parameter with everything else held fixed -- never both at once
amaterasu convergence calc.yaml --scan basis.ecut   --range 20:60:5
amaterasu convergence calc.yaml --scan kpoints.grid --range 2:14:2
```

Any format ASE can read (POSCAR/CONTCAR, CIF, xyz/extxyz, ...) is accepted wherever a
structure file is expected. A run writes `resolved_input.yaml` (every default expanded)
and an HDF5 file holding the energies, eigenvalues, occupations and density in the output
units, with the amaterasu version and the MD5 of every pseudopotential recorded alongside.

`examples/si/` holds runnable inputs for each solver: `scf.yaml` (a single point),
`relax.yaml` (relaxing a displaced atom back onto its Wyckoff site) and `md.yaml` (a short
constant-energy trajectory).

## ASE calculator

`amaterasu.calculator.Amaterasu` is an `ase.calculators.calculator.Calculator`, so anything
in ASE works against it directly:

```python
from ase.build import bulk
from ase.optimize import FIRE
from amaterasu.calculator import Amaterasu
from amaterasu.config import load_config

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = Amaterasu(load_config("calc.yaml"))
FIRE(atoms).run(fmax=0.01)
print(atoms.get_potential_energy(), atoms.get_stress())
```

Between calls the previous density and orbitals are reused as the starting guess whenever
they still fit the new basis, which roughly halves the iterations along a trajectory.
Whether they fit is checked against the assembled system, not guessed from the input:
displacing one atom of a symmetric crystal lowers its space group, which enlarges the
irreducible wedge and changes both the k-point count and the size of every plane-wave
sphere.

## Validation

Phase 2 is checked against data that does not come from amaterasu.

**Free-electron limit.** Setting the pseudopotential to zero must give
`eps = |k+G|^2/2` exactly; it does, to 5e-14 Ha, which tests the basis, the sphere-to-grid
map and the eigensolver against a closed-form answer.

**PseudoDojo equation of state.** The DojoReport publishes ABINIT total energies at listed
volumes computed with the *same* psp8 file, so the pseudopotential, the functional and the
structure are all held fixed and only the plane-wave machinery differs. For silicon at the
dataset's recommended 18 Ha cutoff:

| | V0 (Å³/atom) | B0 (GPa) | B1 |
|---|---|---|---|
| amaterasu | 20.4486 | 88.36 | 4.263 |
| ABINIT (DojoReport) | 20.4467 | 88.18 | 4.282 |

The delta-gauge between the two curves is **0.041 meV/atom**, against the dataset's own
published 0.138 meV/atom versus all-electron WIEN2k -- so amaterasu inherits that figure.
The fitted lattice constant is a₀ = 5.4699 Å (PBE; experiment 5.431 Å) and the indirect
gap is 0.613 eV with the conduction minimum along Γ→X, both as expected for PBE silicon.
The same comparison for aluminium gives V0 within 0.10% and a delta-gauge of
0.29 meV/atom. The residue there is the smearing protocol, which the report does not state
and which a metal is sensitive to. Widening the smearing from 0.014 eV to 0.1 eV shifts
the total energy by half a meV per atom and moves aluminium *towards* the reference --
B0 from +2.6% to +1.1%, and the delta-gauge from 0.29 to 0.23 meV/atom -- so the remaining
difference is a protocol difference rather than a discrepancy in the machinery.

**Ferromagnetic iron.** At the dataset's own PBE equilibrium (a = 2.8419 Å, 45 Ha, a
12×12×12 mesh, 0.1 eV cold smearing) the spin moment is **2.279 μB/atom**, against an
experimental 2.22. The moment is strongly volume-dependent -- 2.17 μB under 7% compression
and 2.32 μB at the experimental 2.87 Å -- so the volume it is quoted at is part of the
number.

**Forces and stress.** Every analytic derivative is checked against a *frozen-orbital*
finite difference -- displace an atom or strain the cell, hold the plane-wave coefficients
fixed, and recompute each energy term. That is the definition of the Hellmann-Feynman
derivative, so it tests each term separately, to machine precision, without a
self-consistent cycle in the loop: forces agree to 2e-7 Ha/Bohr and all six stress
components to 3e-8 GPa. Against *full* self-consistent finite differences on a
low-symmetry cell, the forces agree to 1e-5 eV/Å and sum to zero to 8e-8 eV/Å.

Two things about forces in a plane-wave code are worth stating plainly, because both are
easy to get quietly wrong:

* With smeared occupations the force is the gradient of the **free energy** `F = E - σS`,
  not of `E` and not of the `σ → 0` extrapolation that `get_potential_energy()` returns.
  Differencing the latter looks almost right and is off by half the entropy derivative --
  6 meV/Å on the test cell. `get_potential_energy(force_consistent=True)` gives `F`, and
  that is what the molecular dynamics conserves.
* There is **no Pulay force**: the basis is attached to the cell, not to the atoms. There
  *is* a Pulay stress, because straining the cell changes which plane waves fall under the
  cutoff. It is measured rather than assumed -- on the test cell it is −0.080 GPa at
  24 Ry and −0.0027 GPa at 40 Ry -- and a variable-cell relaxation says so in its log.

**Energy conservation.** A 150 fs constant-energy trajectory of silicon holds its conserved
energy (free energy plus kinetic) to a peak-to-peak 0.08 meV/atom, with a secular drift of
3e-5 meV/atom/ps. The trajectory length is part of that statement: silicon's optical mode
has a 64 fs period, and a drift fitted over less than one period measures the slope of the
oscillation instead -- the same trajectory gives +1.4 meV/atom/ps at 30 fs and
−0.5 meV/atom/ps at 40 fs while the energy itself never moves. `MDResult.energy_spread`
is the quantity that means something at any length.

**Convergence.** Scanned independently, with the other parameter fixed (silicon, PBE):
the cutoff reaches 1 meV/atom at 35 Ry -- the dataset recommends 36 Ry -- and the
Monkhorst-Pack mesh reaches 1 meV/atom at 10×10×10.

**Algorithms against each other.** Density mixing (Pulay, Broyden, Anderson, linear, all
with Kerker preconditioning) and direct minimisation of the energy over the Stiefel
manifold (L-BFGS and conjugate gradient) reach the same ground state to 1e-9 eV. The two
share no code beyond the Hamiltonian, so agreement is a statement that the energy
functional and the operator are consistent.

**Identities checked every iteration**, not only at the end: the electron count, and
`sum_nk w f eps == T + int v_eff n + E_nl`, which ties the eigensolver, the Hamiltonian and
the density accumulation together and holds to machine precision.

### Band structures

A band structure is a *non*-self-consistent calculation, and the thing that can go wrong is
that the frozen density, the rebuilt k-point set and the reassembled potential stop
describing the same Hamiltonian. The check is to feed the self-consistent mesh back in as
if it were a path: every eigenvalue comes back to **1e-9 Hartree**. Along seekpath's
standard path, silicon's smallest gap is 0.608 eV -- the same value the mesh gives, and the
valence band top at Γ keeps its threefold degeneracy to better than 1e-6 eV, which nothing
averages away on a path.

### PAW

**The delta factor** (Lejaeghere et al., Science 351, aad3000 (2016)) is the decisive test
for a PAW implementation, because the reference is a genuinely all-electron calculation
rather than another pseudopotential code. Against WIEN2k, with the JTH v2.0 datasets at
40 Ry / 320 Ry:

| | V0 (Å³/atom) | B0 (GPa) | B1 | Δ (meV/atom) |
|---|---|---|---|---|
| O (8), molecular C2/m, antiferromagnetic | 18.6439 (+0.46%) | 52.62 (+2.4%) | 2.43 | **0.871** |
| Al (13) fcc | 16.4816 (+0.01%) | 80.49 (+3.1%) | 3.62 | **0.200** |
| Si (14) diamond | 20.5513 (+0.48%) | 83.13 (−6.1%) | 4.61 | **1.868** |
| Fe (26) bcc, ferromagnetic | 11.2795 (−0.57%) | 188.33 (−4.7%) | 7.13 | **2.699** |
| Cu (29) fcc | 11.9381 (−0.11%) | 142.45 (+0.8%) | 4.89 | **0.391** |

VASP reports 1.9 and GPAW 3.3 for the same quantity, so all five are in that range. The
PAW and norm-conserving lattice constants for silicon agree to 0.0083 Å (5.4783 vs
5.4700 Å), which is the independent check that the two very different treatments of the
core region describe the same crystal.

`B1` is a third derivative fitted from seven points over ±6%, and oxygen shows what that
is worth — 2.43 against WIEN2k's 3.89, while V0 and B0 stay within 0.5% and 2.4%. The
delta factor, which weights the curve rather than any one coefficient, is the number the
gate is set on.

**Oxygen is the awkward one, and gets checked separately.** Its cell is four atoms in two
O₂ molecules, and the protocol asks for antiferromagnetism *between* them — which presumes
each molecule stays the spin triplet `X ³Σg⁻` it is in the gas phase. That is verified
rather than assumed
([`test_paw_oxygen_spin.py`](tests/physics/test_paw_oxygen_spin.py)):

| | measured |
|---|---|
| one O₂ in a box, total moment | **+2.0000 μB** — the triplet, exactly |
| triplet below the spin-restricted solution by | 1.087 eV/molecule (experiment: `a ¹Δg` at 0.98 eV) |
| crystal, ferromagnetic arrangement, total moment | **+4.0000 μB** — two triplet molecules |
| crystal, lowest of four arrangements | antiferromagnetic, as the protocol prescribes |
| cost of aligning the molecules instead | +55.8 meV/atom |
| cost of opposing the two atoms *within* a molecule | +305.4 meV/atom — it collapses to the non-magnetic solution exactly |

An antiferromagnet's total moment is zero, and so is a non-magnetic solution's, so the
converged total cannot tell them apart; the moments inside the PAW spheres can. Oxygen's
`r_c` is 1.415 Bohr and most of the 2p spin density lies outside it, so a full 1 μB atom
registers as only 0.223 in its sphere — which is what the isolated triplet gives, and what
the ferromagnetic crystal gives. The antiferromagnet's ±0.212 is therefore a full moment,
slightly reduced by intermolecular overlap, and it survives at both ends of the delta
volume range (±0.210 at 94%, ±0.214 at 106%).

**Reading the other formulation.** A Kresse-Joubert dataset carries the local ionic
pseudopotential `v_H[ñ_Zc]` where a Blöchl one carries the zero potential `v̄`, and carries
no kinetic-energy differences at all. Both are reconstructed on load, and both
reconstructions are tested rather than trusted: the synthetic file the unit tests read is
produced by running the same transformation *forwards* on a dataset whose `v̄` is known, so
reading it back is a round trip that returns what went in to **2e-6 Hartree**. The kinetic
differences are checked against a second, independent discretisation of the same integral
(they agree to 3e-5 Hartree) and against the values a PAW-XML file ships.

**PBEsol + PAW, where no PAW-XML dataset exists at all.** The reader's real test is an
equation of state, and PBEsol is the case that needs it: the only PAW datasets for that
functional are Kresse-Joubert UPF files. Silicon, same code and settings, fitted over the
same seven volumes:

| | V0 (Å³/atom) | a₀ (Å) | vs experiment (5.431 Å) |
|---|---|---|---|
| PBEsol PAW, pslibrary kjpaw (UPF) | 20.1020 | 5.4380 | +0.13% |
| PBEsol NC, PseudoDojo (psp8) | 20.0283 | 5.4314 | +0.01% |
| PBE PAW, pslibrary kjpaw (UPF) | 20.4682 | 5.4709 | +0.73% |

The PAW and norm-conserving PBEsol lattice constants agree to **0.0067 Å**, inside the
0.01 Å the phase gate allows — computed from two dataset formats, two PAW formulations and
two independent generators, so nothing about the derived `v̄` or `ΔT_ij` is being taken on
trust. Both land where PBEsol is supposed to land: PBE overestimates silicon by 0.73% and
PBEsol restores the gradient expansion for solids and removes almost all of it.
[`examples/si/paw-pbesol.yaml`](examples/si/paw-pbesol.yaml) is that calculation.

**The dataset's own all-electron eigenvalues.** A PAW-XML file records the eigenvalues of
the atom it was generated from. A silicon atom in a box reproduces the 3p−3s splitting to
2.8 meV (6.7344 against 6.7316 eV) -- one number that exercises the projectors, the
occupancy matrices, the one-centre energy, the compensation charge and the overlap
operator at once, against a value that did not come from amaterasu.

**Known limitation: the compensation charge's shape.** It is a free choice and the total
energy must not depend on it. It does. A Gaussian moves silicon by about 85 meV/atom across
the range of widths that are usable at all, caught between leaking past the augmentation
radius when broad and making the energy functional unbounded below when narrow.

The compactly supported `bessel` shape the datasets themselves declare is implemented and
selectable (`pseudo.compensation_shape: bessel`). It cannot leak, and it cuts the residual
to 26 meV/atom, but it does not remove it — and doubling the density cutoff does not shrink
what is left by more than 0.01 meV, so this is not a basis-set effect. What remains is the
part of the smooth density inside the sphere that the dataset's partial waves cannot span:
the JTH silicon set has two `s` and two `p` channels and no `d`.

The two shapes trade places depending on the element. Six element/dataset combinations
at identical settings, by atomic number:

| Δ (meV/atom), gate < 3 | partial waves | gaussian | bessel |
|---|---|---|---|
| O (8), JTH | s s p p | **0.871** | 2.462 |
| Al (13), JTH | s s p p | **0.200** | 2.048 |
| Si (14), JTH | s s p p | **1.868** | 2.378 |
| Si (14), pslibrary | s s p p d d | **0.326** | 0.658 |
| Fe (26), JTH | s s p p d d | 2.699 | **0.825** |
| Cu (29), JTH | s s p p d d | 0.391 | **0.064** |

Both shapes clear the gate in all six; the Gaussian wins four. The working hypothesis —
that the compact shape should win exactly when the dataset carries `d` partial waves,
since those are what shrink the residual it multiplies — is **refuted** by the pslibrary
silicon row: it has two `d` channels and the Gaussian still wins by a factor of two. What
the rows split on is the element, main group against 3d transition metal, and that is a
description of six points rather than a mechanism.

Where the compact shape does win, it wins on the *curvature* of the equation of state and
against the dataset's own all-electron eigenvalues:

| | gaussian | bessel |
|---|---|---|
| B0 error, Si / Al | −6.1% / +3.09% | **−0.69% / +2.40%** |
| Si lattice constant vs NC (gate 0.01 Å) | **+0.0083 Å** | +0.0105 Å ✗ |
| the same, with the pslibrary dataset | **+0.0009 Å** | +0.0024 Å |
| isolated Si atom, 3p−3s vs the dataset's own AE eigenvalues | 117.7 meV | **68.6 meV** |
| SCF iterations, that atom | 52 | **28** |

The compact shape also costs more density cutoff, because its form factor decays
algebraically rather than exponentially and the tail scales with `1/r_c`. The unresolved
multipole amaterasu reports at run time, at `ecut_rho = 160 Ha`: gaussian 1.9e-05 (O),
4.4e-11 (Si), 2.4e-12 (Cu); bessel 2.5e-04, 2.3e-05, 2.0e-05. Oxygen with `bessel` is the
one case that crosses the 1e-4 the warning fires at; `ecut_rho = 320 Ha` clears it.

**The Gaussian stays the default**: it passes every gate on every element tried and wins
four of the six, and the only gate failure anywhere is the compact shape's +0.0105 Å —
an artefact of that one dataset, since the pslibrary table brings it back to +0.0024 Å, but
a failure all the same. `pseudo.compensation_shape: bessel` is a verified alternative, and
it is the one to reach for on a 3d transition metal.

## Roadmap

| Phase | Content | Validation gate |
|---|---|---|
| 0 ✅ | package, units, structure I/O, YAML schema, CLI, symmetry, k-points, G-vectors, FFT | Si reference numbers; every input error reported in one pass |
| 1 ✅ | LDA/PBE/PBEsol; psp8 + UPF readers | agreement with libxc to 3e-15; pseudopotential invariants |
| 2 ✅ | norm-conserving plane-wave SCF, spin-polarised | empty-lattice test; PseudoDojo Δ-gauge 0.04 meV/atom for Si; Fe moment 2.28 μB |
| 3 ✅ | forces, stress, ASE calculator, relaxation, MD | forces to 1e-5 eV/Å and stress to 3e-8 GPa against finite differences; ΣF = 8e-8 eV/Å; NVE conserved to 0.08 meV/atom |
| 4 ✅ | PAW: PAW-XML *and* UPF (Kresse-Joubert) readers, partial waves, compensation charges, D_ij, on-site XC, overlap operator, forces and stress | Δ-factor against WIEN2k for five elements, all inside the 3 meV/atom gate: **O 0.87, Al 0.20, Si 1.87, Fe 2.70, Cu 0.39**; PAW and NC lattice constants agree to 0.0083 Å for PBE and 0.0067 Å for PBEsol. The compensation-charge shape dependence is measured, bounded and documented, and the Gaussian is the settled default |
| 5 ◐ | bands, DOS/PDOS, CI-NEB, phonons, DFT+U | band structure lands first: silicon's indirect gap along the seekpath path is **0.608 eV**, the same number the self-consistent mesh gives. Outstanding: DOS/PDOS, NEB, phonons, DFT+U |
| 6 | DFPT: dielectric, Born charges, phonons, elastic | DFPT phonons ≡ finite-displacement phonons; ΣZ* = 0 |
| 7 | Wannier functions (Wannier90 files, then native MLWF) | spreads match Wannier90 on identical input |
| 8 | mixed basis (plane waves + atomic orbitals) | converges to the pure plane-wave result |
| 9 | numba kernels, MPI over k-points/bands/plane waves | results independent of process count |

## Pseudopotentials and PAW datasets

Dataset files are not bundled (their licences differ from this project's); they are
fetched on demand. Recommended sources:

| functional | norm-conserving | PAW |
|---|---|---|
| PBE | [PseudoDojo](http://www.pseudo-dojo.org/) ONCVPSP (CC BY 4.0) | [JTH v2.0](https://www.abinit.org/atomic_data/paw/) PAW-XML (GPL-3.0) |
| PBEsol | PseudoDojo ONCVPSP | [pslibrary](https://dalcorso.github.io/pslibrary/) `*.pbesol-*-kjpaw_psl.*.UPF` |

Note that PBEsol PAW datasets exist **only** in UPF form: the PAW-XML tables (JTH, GPAW)
ship PBE and LDA only. Those UPF files are written in the *Kresse-Joubert* formulation of
PAW rather than Blöchl's, and two of the quantities the rest of the code needs are not in
them; amaterasu derives both on load (see
[`upfpaw.py`](amaterasu/pseudo/readers/upfpaw.py)). They are not fetchable by
`fetch-pseudos` — pslibrary's filenames vary per element — so download one and name it:

```yaml
pseudo:
  method: paw
  files: {Si: Si.pbesol-n-kjpaw_psl.1.0.0.UPF}
```

amaterasu refuses by default to use a dataset generated with a functional other than the
one requested.

## Testing

```bash
pip install -e ".[test]"
pytest                 # fast unit tests and doctests
pytest -m physics      # slow validation against reference physical data
ruff check . && ruff format --check .
```

### libxc, for exchange-correlation validation

amaterasu implements LDA, PBE and PBEsol itself and checks them point-by-point against
libxc. libxc's Python bindings are **not on PyPI** -- the `pylibxc` and `pylibxc2` names
there are unrelated stubs -- so build from source:

```bash
curl -L -o libxc.tar.gz https://gitlab.com/libxc/libxc/-/archive/7.1.2/libxc-7.1.2.tar.gz
tar xzf libxc.tar.gz && cd libxc-7.1.2
pip install .
```

No system CMake is needed; pip fetches it into an isolated build environment. Tests that
need libxc skip when it is absent, so this step is optional.

Two details this comparison surfaced, which amaterasu's own implementation follows:

* libxc's `gga_c_pbe` reduces at zero gradient to `lda_c_pw_mod`, not `lda_c_pw` -- the
  PW92 parametrisation with the extra decimal places, as used in the PBE paper. Building
  PBE on plain `lda_c_pw` leaves a systematic 2.5e-7 Ha/electron offset.
* The gradient correction must therefore be measured as `eps_c(sigma) - eps_c(0)` of the
  same functional; subtracting `lda_c_pw` leaves that offset and the extracted `beta`
  diverges instead of converging.

The PBE and PBEsol parameters used by amaterasu were confirmed numerically against
libxc 7.1.2 rather than quoted from the papers:

| | mu | beta | kappa |
|---|---|---|---|
| PBE | 0.2195149728 | 0.0667245506 | 0.804 |
| PBEsol | 0.1234567901 (= 10/81) | 0.046 | 0.804 |

## License

Apache-2.0 -- see [LICENSE](LICENSE).
