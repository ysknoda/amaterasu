---
status: accepted (amended by 0008)
date: 2026-09-15
---

# The primary gate for orbital-free DFT is amaterasu's own Kohn-Sham run with the same local pseudopotential; literature values are the secondary gate; aluminium first, silicon as a measured-failure study

> **Amended by [ADR-0008](0008-nonlocal-pseudopotentials-via-the-density-matrix.md).**
> The gate structure below stands and gets stronger: Kohn-Sham and orbital-free now
> consume the identical file *including its projectors*, so the difference really is
> the functional alone. The pseudopotential source and the element list are replaced
> by the table in ADR-0008.

An orbital-free result differs from a Kohn-Sham one for three separable reasons: the
kinetic functional, the pseudopotential, and the numerics (grid, cutoff, minimiser). The
obvious comparison -- against published PROFESS numbers for aluminium -- mixes all three.
Because this code also runs Kohn-Sham DFT, it can hold the pseudopotential and the grid
fixed and isolate the functional: run KS and OF on the *same* BLPS, same cutoff, same
cell, and the difference is the KEDF alone. That is the primary gate.

Published values (Carter group, aluminium and silicon equations of state, vacancy
formation energies) are the secondary gate, and the tables are obtained from the user as
source documents when needed.

Targets, in order:

- **Aluminium fcc** -- nearly free electrons, where nonlocal KEDFs are known to work.
  The whole ladder is run here and each rung's error against KS is tabulated.
- **Silicon diamond** -- covalent, where semilocal functionals and Wang-Teter/WGC are
  known to fail. Silicon is in scope *to measure and record that failure* with the same
  instrument, not to pass a gate. Whether to implement the Huang-Carter functional, which
  targets semiconductors but is far costlier and harder to converge, is decided from the
  measured silicon numbers, not in advance.
- Magnesium is dropped: same physics as aluminium, nothing new learned.
