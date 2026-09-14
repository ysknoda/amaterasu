# Test pseudopotentials

These files are redistributed here as test fixtures only. amaterasu does not bundle
pseudopotentials for production use; those are fetched on demand (see the README).

## PseudoDojo ONCVPSP (`*.psp8.gz`)

| file | table | functional |
|---|---|---|
| `Si.pbe.psp8.gz` | ONCVPSP-PBE-PDv0.4 | PBE |
| `Si.pbesol.psp8.gz` | ONCVPSP-PBEsol-PDv0.4 | PBEsol |
| `O.pbe.psp8.gz` | ONCVPSP-PBE-PDv0.4 | PBE |

Source: <http://www.pseudo-dojo.org/> and <https://github.com/abinit/pseudo_dojo>.
The pseudopotential files are licensed **CC BY 4.0**, which permits redistribution with
attribution.

Please cite:

> M. J. van Setten, M. Giantomassi, E. Bousquet, M. J. Verstraete, D. R. Hamann,
> X. Gonze and G.-M. Rignanese, "The PseudoDojo: Training and grading a 85 element
> optimized norm-conserving pseudopotential table",
> Computer Physics Communications 226, 39 (2018).

Files are stored gzip-compressed (the readers decompress `.gz` transparently); they are
byte-for-byte the upstream files otherwise.

## UPF

No UPF file is redistributed here. The UPF reader is exercised against a synthetic file
constructed in `tests/unit/test_pseudo_readers.py`, which covers every branch of the
parser, and against real files when a directory of them is supplied via the
`AMATERASU_TEST_PSEUDO_DIR` environment variable.
