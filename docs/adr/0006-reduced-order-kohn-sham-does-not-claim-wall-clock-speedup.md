---
status: accepted
date: 2026-09-15
---

# The reduced-order Kohn-Sham track is gated on accuracy and basis size, not on wall-clock time, because Galerkin projection alone does not reduce the cost of applying the Hamiltonian

Projecting the Kohn-Sham problem onto a POD basis of `M` orbitals turns the eigenproblem
into an `M x M` one, but every application of `H` to a basis vector is still a full-grid
FFT: the kinetic and local terms act in the plane-wave basis, and the Hartree and XC
potentials are rebuilt from the density on the grid. Without an *affine decomposition* of
`H(R)` into structure-independent operators with structure-dependent scalar
coefficients -- the offline/online split that makes reduced-order models fast in
engineering -- the projection saves the eigensolver's cost and nothing else. That
decomposition is a project of its own and is out of Phase 10.

Decision: the track's gates are (1) the error against the full Kohn-Sham solve at fixed
`M`, and (2) `M` being much smaller than the number of plane waves, both measured on
frozen-phonon displacements of silicon in a fixed cell at the Gamma point. Wall-clock time
is reported, never gated. The reduced solve is a full self-consistent cycle in the reduced
space -- the density is rebuilt from the reduced eigenvectors every iteration -- not a
one-shot projection, because a one-shot projection is a test of the basis, not a
ground-state solve.

The value of the track, stated honestly, is a measurement of how many orbitals a
parameter sweep actually needs, and a working reduced eigensolver on which an affine
decomposition could later be built. It is not a faster DFT yet, and the code must not say
it is.
