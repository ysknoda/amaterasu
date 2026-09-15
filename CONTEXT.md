# amaterasu

A plane-wave density functional theory code for three-dimensional periodic crystals. This
file is the glossary: what the project's words mean, and which near-synonyms to avoid.
It carries no implementation detail.

## Language

### Electronic-structure engines

**Kohn-Sham theory**:
The default engine: the ground state is found by solving for one-electron orbitals whose
density minimises the energy. Has eigenvalues, occupations, a Fermi level and k-points.
_Avoid_: "standard DFT", "normal DFT" (say which engine)

**Orbital-free theory**:
The engine that finds the ground state by minimising the energy directly over the
electron density, with no orbitals, no k-points and no eigenvalues. Requires a kinetic
energy density functional. It does *not* require a local pseudopotential: a nonlocal one
is used through the nonlocal pseudopotential functional below.
_Avoid_: OF-DFT as a noun in prose (fine as an abbreviation in code and tables)

**Ground-state solve**:
One run of either engine from a structure to a converged density and energy. In
Kohn-Sham theory it is a self-consistent field cycle; in orbital-free theory it is a
direct minimisation, and there is no "field" to make self-consistent.
_Avoid_: "SCF" for the orbital-free case

**Kinetic energy density functional (KEDF)**:
The approximation to the non-interacting kinetic energy as a functional of the density
alone, which orbital-free theory needs and Kohn-Sham theory does not. Named by its
functional form (Thomas-Fermi, von Weizsacker, LKT, Wang-Teter, Wang-Govind-Carter,
Huang-Carter), never by who coded it.
_Avoid_: "kinetic functional" without "density" (ambiguous with the exact orbital form)

**Nonlocal pseudopotential functional**:
The expression that gives the nonlocal pseudopotential energy without orbitals, by
integrating the projectors against a density matrix that is itself a functional of the
density. The thing that lets orbital-free theory use an ordinary norm-conserving file.
_Avoid_: `NLPPF` in prose (fine when citing the source paper); the identifier everywhere
in code and config is `nonlocal_pseudo`

**Density matrix**:
`gamma(r, r')`, the non-interacting one-particle density matrix. In Kohn-Sham theory it
is built from orbitals; in orbital-free theory it is approximated as a functional of the
density (modified Gaussian). Say "off-diagonal" for the `r != r'` part.
_Avoid_: calling its off-diagonal character "nonlocality" -- that word already means two
other things here (see below)

**Kinetic energy kernel**:
`K_s(q)`, the second functional derivative of the kinetic energy with respect to the
density. What every two-point KEDF is built from, and a thing this code *measures* from
its own Kohn-Sham runs rather than only assuming.

**Calibration**:
Fitting the two density-matrix parameters for one element against a Kohn-Sham reference,
once, and storing the result beside the pseudopotential. The same kind of act as
generating a pseudopotential, and reported separately from accuracy evidence.
_Avoid_: using "fitting" for this without saying what the target was

**Reduced basis**:
A small set of orthonormal functions, built by SVD (proper orthogonal decomposition) of
Kohn-Sham orbitals from earlier ground-state solves, in which a later Kohn-Sham problem is
solved by Galerkin projection. Fixed once built; valid only for structures near the ones
it was built from.
_Avoid_: latent basis, latent space (those words imply learning)

**Snapshot**:
One converged set of Kohn-Sham orbitals at one structure, kept as input to building a
reduced basis.

**Rank survey**:
A measurement, not a method: the tensor-train rank a grid quantity (density, orbital,
kernel) needs to reach a given accuracy. It informs whether tensor-train arithmetic is
worth implementing; it does not itself speed anything up.

### Pseudopotentials and datasets

**Dataset**:
A PAW description of one element: partial waves, projectors, augmentation data. Read
from PAW-XML or Kresse-Joubert UPF.
_Avoid_: calling a PAW file a "pseudopotential"

**Pseudopotential**:
A norm-conserving description of one element: a local potential plus Kleinman-Bylander
nonlocal projectors. Read from psp8 or UPF.

**Local pseudopotential**:
A pseudopotential consisting of a local potential only, with no nonlocal projectors.
Bulk-derived local pseudopotentials (BLPS) are the example; they are no longer the
production source for orbital-free theory, but they are its zero-projector reference
case, on which the nonlocal pseudopotential functional must return exactly zero.
_Avoid_: "LPS" in prose; "local part" (that means the local component *of* a nonlocal file)

**Evidence class**:
What this project has actually measured about one element: `validated` (passed our own
gate), `expected` (no d or f projectors, untested), `unvalidated` (has them). Never set
from someone else's published results, and never merged with calibration state -- an
element can be calibrated and still unvalidated.

**Local stub**:
The local part of a norm-conserving pseudopotential with its projectors discarded, used
only to develop and test orbital-free code before real local pseudopotentials are
available. Not a pseudopotential; every run on one must say so.

### Compensation charge (PAW)

**Compensation shape**:
The fixed radial function that carries a PAW atom's multipole moments on the plane-wave
grid. `gaussian` (default) or `bessel` (compact support). A free choice the total energy
must not, but measurably does, depend on.
_Avoid_: shape function (too generic), augmentation shape

### The three meanings of "nonlocal"

The word is load-bearing in three unrelated places. Always qualify it:

- **nonlocal projector** -- the Kleinman-Bylander part of a pseudopotential. Opposite of
  a local pseudopotential.
- **nonlocal KEDF** -- a kinetic energy density functional with a two-point kernel
  (Wang-Teter, WGC, Huang-Carter). "Two-point KEDF" is an acceptable synonym.
- the **off-diagonal** density matrix -- say that, not "nonlocal".
