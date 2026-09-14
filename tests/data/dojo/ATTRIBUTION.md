# PseudoDojo equation-of-state reference data

`deltafactor.json` holds an excerpt of the DojoReport (`*.djrepo`) files published with
the PseudoDojo pseudopotential tables. For each element it records the ABINIT total
energies at a listed set of volumes -- computed with the *same* psp8 file bundled under
`tests/data/pseudos/`, at the stated cutoff -- together with the Birch-Murnaghan
parameters fitted to them and the published delta factor against the WIEN2k all-electron
reference.

Holding the dataset, the functional and the structure fixed is what makes this a test of
amaterasu's plane-wave machinery rather than of the pseudopotential.

Source: <https://github.com/abinit/pseudo_dojo>, `pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.4/`.
Same provenance and licence (**CC BY 4.0**) as the psp8 files themselves; see
`tests/data/pseudos/ATTRIBUTION.md` for the citation.

The delta factor itself is defined in:

> K. Lejaeghere et al., "Reproducibility in density functional theory calculations of
> solids", Science 351, aad3000 (2016).
