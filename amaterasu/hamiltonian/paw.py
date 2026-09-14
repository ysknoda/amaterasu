"""What PAW adds to the plane-wave side of the calculation.

Three things change relative to a norm-conserving run, and each one is here:

**The charge density gains a compensation charge.** ``rho~ = n~ + sum_a Z~^a``, where
``Z~^a`` carries exactly the multipole moments that the smooth density is missing inside
atom ``a``. The point of it is that ``rho~`` is then *neutral* -- the nuclear charge is
inside it -- so the plane-wave Hartree energy with ``G = 0`` dropped is the entire
electrostatic energy of the crystal, ion-ion interaction included. There is no Ewald sum
in a PAW calculation, and no local ionic pseudopotential either: what is left of the
latter is the short-ranged ``zero potential`` ``v_bar``, which vanishes outside the
augmentation sphere.

**The orbitals are no longer orthonormal.** The smooth orbitals satisfy
``<psi~|S|psi~> = delta`` with ``S = 1 + sum_a sum_ij |p_i> O_ij <p_j|``, so the
eigenproblem is generalised. ``O_ij`` is the amount of charge the smooth orbitals are
missing inside the sphere, and it is the same object as the ``l = 0`` multipole of the
augmentation, ``sqrt(4 pi) Delta_00ij``.

**The nonlocal coefficients are not constants.** Where a norm-conserving ``D_ij`` comes
from the file and never changes, the PAW one is ``dE/dD_ij`` and has to be rebuilt every
iteration from the current potential and the current occupancies:

    H^a_ij = [ d(E^a - E~^a)/dD_ij ]  +  sum_L Delta^a_Lij W^a_L ,
    W^a_L  = integral v~_H(r) g^a_L(r - R_a) d^3r ,

the first term computed on the radial mesh by :mod:`amaterasu.pseudo.paw.onsite` and the
second on the plane-wave grid here. ``W^a_L`` is the smooth Hartree potential's projection
onto the shape function, and it is what couples the two halves of the calculation.

Compensation charges on the grid
--------------------------------
Both shape functions have closed-form Fourier transforms, so the compensation charge
lands on the grid with no quadrature error:

    Z~^a(G) = (1/Omega) sum_L Q^a_L 4 pi (-i)^l Y_L(G_hat) g~_l(|G|) exp(-i G.R_a) .

The compensation charge is the sharpest object in the density, so it -- not the orbitals
-- is what sets the density cutoff a PAW calculation needs, which is why PAW codes carry a
second, larger cutoff for the density grid. :func:`build_paw_machinery` measures how much of
the monopole is still unresolved at the cutoff actually in use, and how deep a well the
shape digs, and says so if either is out of range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..basis.fft import FFTGrid
from ..basis.gvectors import GSphere
from ..pseudo.formfactors import radial_transform
from ..pseudo.kb import KBProjectors
from ..pseudo.paw.angular import SphericalGrid, real_harmonics
from ..pseudo.paw.dataset import PAWDataset
from ..pseudo.paw.onsite import OnsiteResult, OnsiteSetup
from ..pseudo.paw.shape import COLLAPSE_ALPHA_RC2, equivalent_alpha_rc2, make_shape
from ..system.symmetry import SymmetryInfo, atom_mapping, cartesian_operations
from .local import structure_factors

__all__ = [
    "PAWMachinery",
    "build_paw_machinery",
    "build_paw_projectors",
    "harmonic_rotation",
    "vbar_potential",
]


def harmonic_rotation(lmax: int, operation: np.ndarray) -> np.ndarray:
    """``W_LL'`` with ``Y_L(R n) = sum_L' W_LL' Y_L'(n)``, for ``l = 0 .. lmax``.

    Computed by quadrature rather than from a Wigner-D recursion: the integrand
    ``Y_L(R n) Y_L'(n)`` is a polynomial of degree ``2 lmax`` on the sphere, so a rule
    exact to that degree gives the matrix to machine precision -- and the quadrature grid
    is already there for the exchange-correlation integration. Improper operations need no
    special treatment; the harmonics are simply evaluated at the reflected direction.

    ``operation`` is the row-vector Cartesian matrix of
    :func:`amaterasu.system.symmetry.cartesian_operations`, so ``directions @ operation``
    is the rotated set of directions.
    """
    grid = SphericalGrid.build(2 * lmax)
    rotated = real_harmonics(lmax, grid.directions @ np.asarray(operation, dtype=float))
    return np.einsum("k,ik,jk->ij", grid.weights, rotated, grid.harmonics(lmax))


def _paw_projector_form_factors(setup: OnsiteSetup, q: np.ndarray) -> np.ndarray:
    """``integral p~_nl(r) j_l(q r) r^2 dr`` for every radial channel, shape ``(nchannels, nq)``.

    PAW-XML stores the plain radial projector ``p~(r)``, where a norm-conserving file
    stores ``r beta(r)``; the ``r^2`` of the volume element is supplied here rather than
    being half-hidden in the stored convention.
    """
    data = setup.dataset
    stop = setup.stop
    weight = np.zeros(len(data.grid))
    factors = []
    for wave in data.waves:
        weight[:] = 0.0
        weight[:stop] = wave.projector[:stop] * data.grid.r[:stop] ** 2
        factors.append(radial_transform(data.grid, weight, wave.l, q))
    return np.array(factors)


def build_paw_projectors(
    sphere: GSphere,
    positions: np.ndarray,
    symbols: list[str],
    setups: dict[str, OnsiteSetup],
    volume: float,
) -> KBProjectors:
    """``<k+G|p~^a_i>`` for every atom and projector at one k-point.

    The coupling matrix is left at zero: unlike a norm-conserving ``D_ij`` it is not a
    property of the dataset but of the current potential, and
    :meth:`PAWMachinery.coupling` fills it in once per SCF iteration.
    """
    kg = sphere.cartesian
    q = np.sqrt(sphere.g2)
    prefactor = 4.0 * np.pi / np.sqrt(volume)

    form_factors = {symbol: _paw_projector_form_factors(s, q) for symbol, s in setups.items()}
    harmonics: dict[int, np.ndarray] = {}
    for setup in setups.values():
        for wave in setup.dataset.waves:
            if wave.l not in harmonics:
                harmonics[wave.l] = real_harmonics(wave.l, kg)[wave.l**2 :]

    rows: list[np.ndarray] = []
    labels: list[tuple[int, int, int, int]] = []
    for atom, symbol in enumerate(symbols):
        setup = setups[symbol]
        phase = np.exp(-1j * (kg @ positions[atom]))
        for channel, wave in enumerate(setup.dataset.waves):
            radial = form_factors[symbol][channel]
            angular = harmonics[wave.l]
            for offset in range(2 * wave.l + 1):
                rows.append(prefactor * (-1j) ** wave.l * angular[offset] * radial * phase)
                labels.append((atom, wave.l, channel, offset - wave.l))
    size = len(rows)
    return KBProjectors(
        matrix=np.asarray(rows) if rows else np.zeros((0, len(sphere)), dtype=complex),
        dij=np.zeros((size, size)),
        labels=labels,
    )


def vbar_potential(
    grid: FFTGrid,
    sphere: GSphere,
    positions: np.ndarray,
    symbols: list[str],
    setups: dict[str, OnsiteSetup],
) -> np.ndarray:
    """``sum_a v_bar^a(|r - R_a|)`` on the real-space grid, in Hartree.

    This is all that survives of the local ionic pseudopotential in the Blochl
    formulation. It vanishes identically outside the augmentation sphere -- checked, not
    assumed -- so it carries no Coulomb tail and no ``G = 0`` subtlety.
    """
    q = np.sqrt(sphere.g2)
    radial = {}
    for symbol, setup in setups.items():
        data = setup.dataset
        values = np.zeros(len(data.grid))
        stop = setup.stop
        values[:stop] = np.sqrt(4.0 * np.pi) * data.zero_potential[:stop] * data.grid.r[:stop] ** 2
        radial[symbol] = radial_transform(data.grid, values, 0, q)

    phases = structure_factors(sphere, positions)
    coefficients = np.zeros(len(sphere), dtype=complex)
    for atom, symbol in enumerate(symbols):
        coefficients += radial[symbol] * phases[atom]
    coefficients /= grid.volume
    field = grid.to_real(grid.scatter(coefficients, sphere))
    residual = float(np.max(np.abs(field.imag)))
    scale = max(float(np.max(np.abs(field.real))), 1e-30)
    if residual > 1e-8 * scale:  # pragma: no cover - defensive
        raise ValueError(f"the zero potential came out complex ({residual:.2e} vs {scale:.2e})")
    return np.ascontiguousarray(field.real)


@dataclass
class PAWMachinery:
    """The PAW-specific state of one geometry: projector layout, shapes, overlaps."""

    setups: dict[str, OnsiteSetup]
    symbols: list[str]
    #: ``(start, size)`` of each atom's block of rows in the projector matrix.
    blocks: list[tuple[int, int]]
    #: Block-diagonal ``O_ij`` for the whole cell, shape ``(nproj, nproj)``.
    overlap: np.ndarray
    #: ``Z~^a_L(G)`` per atom: a list of ``(n_lm_a, npw)`` complex arrays on the density
    #: sphere, already carrying the ``1/Omega`` of the transform convention.
    shape_coefficients: list[np.ndarray]
    volume: float
    #: ``(rotations, mapping)`` for symmetrising the occupancies, or ``None`` when the
    #: Brillouin-zone sum runs over the full zone and no average is needed.
    rotations: tuple[list[list[np.ndarray]], np.ndarray] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_projectors(self) -> int:
        return sum(size for _, size in self.blocks)

    def __str__(self) -> str:
        species = ", ".join(
            f"{symbol}: {s.n_projectors} projectors, l<={s.lmax_density} density"
            for symbol, s in self.setups.items()
        )
        return f"PAW ({species})"

    # ------------------------------------------------------------------ occupancies
    def occupancies(self, projections: np.ndarray) -> list[np.ndarray]:
        """Split a cell-wide ``D_ij`` into one matrix per atom.

        ``projections`` has shape ``(nspin, nproj, nproj)``.
        """
        return [
            np.ascontiguousarray(projections[:, start : start + size, start : start + size])
            for start, size in self.blocks
        ]

    def scatter(self, per_atom: list[np.ndarray], nspin: int) -> np.ndarray:
        """Assemble per-atom matrices back into the cell-wide block-diagonal form."""
        total = np.zeros((nspin, self.n_projectors, self.n_projectors))
        for (start, size), block in zip(self.blocks, per_atom, strict=True):
            total[:, start : start + size, start : start + size] = block
        return total

    # ------------------------------------------------------- compensation charge
    def compensation_charge(self, multipoles: list[np.ndarray]) -> np.ndarray:
        """``Z~(G)`` on the density sphere from each atom's multipole moments."""
        coefficients = np.zeros(self.shape_coefficients[0].shape[1], dtype=complex)
        for moments, table in zip(multipoles, self.shape_coefficients, strict=True):
            coefficients += moments @ table
        return coefficients

    def shape_projections(self, potential_g: np.ndarray) -> list[np.ndarray]:
        """``W^a_L = integral v(r) g^a_L(r - R_a) d^3r`` for a potential on the sphere.

        Real by construction: ``v`` and every ``g^a_L`` are real fields, so the sum over
        ``+G`` and ``-G`` pairs is a sum of conjugates. The imaginary part left over is a
        check on the sphere bookkeeping and is asserted to be negligible.
        """
        results = []
        for table in self.shape_coefficients:
            value = self.volume * (table.conj() @ potential_g)
            residual = float(np.max(np.abs(value.imag)))
            scale = max(float(np.max(np.abs(value.real))), 1e-30)
            if residual > 1e-6 * scale and residual > 1e-12:  # pragma: no cover - defensive
                raise ValueError(
                    f"the shape projection W_L came out complex ({residual:.2e} against "
                    f"{scale:.2e}); a phase convention is wrong"
                )
            results.append(value.real)
        return results

    def shape_projection_gradients(self, potential_g: np.ndarray, gvectors: np.ndarray):
        """``d/dR_a integral v(r) g^a_L(r - R_a)``, shape ``(natoms, n_lm_a, 3)``.

        The shape function carries the atom's phase ``exp(-i G.R_a)``, so displacing the
        atom multiplies each coefficient by ``-i G``; the derivative costs one extra
        contraction over the same table.
        """
        results = []
        for table in self.shape_coefficients:
            weighted = self.volume * (table.conj() * potential_g[None, :])
            results.append(np.einsum("Lg,gc->Lc", weighted, 1j * gvectors).real)
        return results

    # ------------------------------------------------------------------ the coupling
    def coupling(
        self, onsite: list[OnsiteResult], potential_g: np.ndarray, nspin: int
    ) -> np.ndarray:
        """``H^a_ij``, block-diagonal over atoms, shape ``(nspin, nproj, nproj)``.

        Equation (73) of Mortensen et al.: the one-centre gradient plus the coupling of
        the compensation charge to the smooth Hartree potential.
        """
        projections = self.shape_projections(potential_g)
        blocks = []
        for atom, (symbol, result) in enumerate(zip(self.symbols, onsite, strict=True)):
            setup = self.setups[symbol]
            extra = np.einsum("L,Lij->ij", projections[atom], setup.delta)
            blocks.append(result.hamiltonian + extra[None, :, :])
        return self.scatter(blocks, nspin)

    def symmetrize(self, occupancies: np.ndarray) -> np.ndarray:
        """Average the occupancy matrices over the space group.

        The same argument as for the density and the forces. A Brillouin-zone sum over the
        irreducible wedge gives a ``D^a_ij`` with the symmetry of the wedge rather than of
        the crystal, and the one-centre energy is *not* a scalar function of it: an
        occupancy matrix that should be spherical acquires a spurious quadrupole, and with
        it a spurious compensation charge and a spurious energy.

        A rotation sends the projections of one atom onto those of its image,
        ``P^b_i = sum_j W_ij P^a_j``, so ``D^b = W D^a W^T`` and the average is
        ``D^a <- (1/N) sum_S W_S^T D^{S(a)} W_S``.
        """
        if self.rotations is None:
            return occupancies
        rotations, mapping = self.rotations
        averaged = np.zeros_like(occupancies)
        for index, matrices in enumerate(rotations):
            for atom, (start, size) in enumerate(self.blocks):
                image = mapping[index, atom]
                source, extent = self.blocks[image]
                block = occupancies[:, source : source + extent, source : source + extent]
                rotation = matrices[atom]
                averaged[:, start : start + size, start : start + size] += np.einsum(
                    "ki,skl,lj->sij", rotation, block, rotation
                )
        return averaged / len(rotations)

    def apply_overlap(self, coefficients: np.ndarray, projectors: KBProjectors) -> np.ndarray:
        """``S |psi~>`` for a block of bands at one k-point."""
        if self.n_projectors == 0:  # pragma: no cover - a cell with no projectors
            return np.asarray(coefficients)
        return coefficients + (projectors.project(coefficients) @ self.overlap) @ projectors.matrix


def _projector_rotations(
    setups: dict[str, OnsiteSetup], symbols: list[str], operations: np.ndarray
) -> list[list[np.ndarray]]:
    """The rotation matrix in projector space for every operation and every atom.

    A rotation mixes the ``m`` values of one shell and leaves the radial channel alone, so
    the matrix is block diagonal in ``(n, l)`` and is just the harmonic rotation lifted to
    the projector index.
    """
    lmax = max(setup.lmax_wave for setup in setups.values())
    per_species: dict[str, list[np.ndarray]] = {symbol: [] for symbol in setups}
    for operation in operations:
        harmonic = harmonic_rotation(lmax, operation)
        for symbol, setup in setups.items():
            same_channel = setup.channel_of[:, None] == setup.channel_of[None, :]
            block = harmonic[np.ix_(setup.lm_of, setup.lm_of)] * same_channel
            per_species[symbol].append(block)
    return [[per_species[symbol][index] for symbol in symbols] for index in range(len(operations))]


def build_paw_machinery(
    grid: FFTGrid,
    sphere: GSphere,
    positions: np.ndarray,
    symbols: list[str],
    setups: dict[str, OnsiteSetup],
    *,
    atoms=None,
    symmetry: SymmetryInfo | None = None,
    cell: np.ndarray | None = None,
) -> PAWMachinery:
    """Assemble the per-geometry PAW tables: projector layout, overlaps, shape transforms."""
    blocks, overlap_blocks = [], []
    offset = 0
    for symbol in symbols:
        size = setups[symbol].n_projectors
        blocks.append((offset, size))
        overlap_blocks.append(setups[symbol].overlap)
        offset += size
    overlap = np.zeros((offset, offset))
    for (start, size), block in zip(blocks, overlap_blocks, strict=True):
        overlap[start : start + size, start : start + size] = block

    gvectors = sphere.cartesian
    q = np.sqrt(sphere.g2)
    phases = structure_factors(sphere, positions)
    lmax = max(s.lmax_density for s in setups.values())
    harmonics = real_harmonics(lmax, gvectors)

    shape_coefficients = []
    for atom, symbol in enumerate(symbols):
        setup = setups[symbol]
        table = np.empty((setup.n_lm, len(sphere)), dtype=complex)
        for index in range(setup.n_lm):
            l = int(np.sqrt(index))
            table[index] = (
                4.0
                * np.pi
                * (-1j) ** l
                * harmonics[index]
                * setup.shape.form_factor(l, q)
                * phases[atom]
                / grid.volume
            )
        shape_coefficients.append(table)

    warnings = []
    qmax = float(np.sqrt(sphere.g2.max())) if len(sphere) else 0.0
    for symbol, setup in setups.items():
        # Too wide. The one-centre integrals run out to the shape's containment radius
        # rather than stopping at r_c, so the tail's *own* energy is subtracted again; what
        # is left is that the tail sits outside the sphere, where the one-centre expansion
        # says nothing about the neighbours it overlaps. That is the leading term in the
        # residual shape dependence -- see the table in pseudo/paw/shape.py. A compactly
        # supported shape leaks exactly nothing, so this fires only for a broad Gaussian.
        escaped = setup.shape.escaped_charge()
        if escaped > 5e-3:
            warnings.append(
                f"{symbol}: {escaped:.1%} of the compensation charge lies outside the "
                f"augmentation sphere, which is where the total energy starts to depend on "
                f"a shape function it should be independent of; narrow it with "
                f"pseudo.compensation_scale, or use the compactly supported "
                f"pseudo.compensation_shape: bessel, which cannot leak at all"
            )
        # Too concentrated. The compensation charge is negative and carries the nucleus,
        # so past some depth the smooth orbitals fall into its own well and the energy has
        # no lower bound; this is the diagnostic that predicts it for any shape.
        depth = equivalent_alpha_rc2(setup.shape, setup.dataset.rcut)
        if depth > COLLAPSE_ALPHA_RC2:
            warnings.append(
                f"{symbol}: the compensation charge digs a well as deep as a Gaussian of "
                f"alpha*r_c^2 = {depth:.1f}, past the {COLLAPSE_ALPHA_RC2:g} where the total "
                f"energy stops being bounded below; broaden it with pseudo.compensation_scale"
            )
        # Too sharp for the grid: the same charge then does not fit under the density
        # cutoff.
        channel, unresolved = _unresolved_multipole(setup.shape, setup.lmax_density, qmax)
        if unresolved > 1e-4:
            warnings.append(
                f"{symbol}: the compensation charge still carries {unresolved:.2e} of its "
                f"l = {channel} multipole at the largest |G| of the density cutoff; "
                f"raise basis.ecut_rho"
            )
    rotations = None
    if symmetry is not None and symmetry.n_operations > 1 and atoms is not None:
        operations = cartesian_operations(symmetry, cell)
        rotations = (
            _projector_rotations(setups, symbols, operations),
            atom_mapping(atoms, symmetry),
        )
    return PAWMachinery(
        setups=setups,
        symbols=list(symbols),
        blocks=blocks,
        overlap=overlap,
        shape_coefficients=shape_coefficients,
        volume=grid.volume,
        rotations=rotations,
        warnings=warnings,
    )


def _unresolved_multipole(shape, lmax: int, qmax: float) -> tuple[int, float]:
    """The multipole the density grid resolves worst, and by how much.

    Not the monopole alone: the higher multipoles of a compact shape are the sharper ones
    and are always the first to run out of grid, so reporting ``l = 0`` would understate
    the problem -- for oxygen's small augmentation radius the ``l = 2`` component is four
    times less resolved than the monopole.
    """
    if qmax <= 0:
        return 0, 0.0
    radius = getattr(shape, "rcut", 0.0) or 1.0
    # A compact shape's form factor oscillates as it decays, so a single point can land on
    # a zero and look converged; take the local envelope over the last full oscillation,
    # whose period in |G| is pi / the shape's own radius.
    window = np.linspace(max(0.0, qmax - np.pi / radius), qmax, 128)
    worst = [float(np.max(np.abs(shape.form_factor(l, window)))) for l in range(lmax + 1)]
    index = int(np.argmax(worst))
    return index, worst[index]


def make_setups(
    datasets: dict[str, PAWDataset],
    xc,
    *,
    shape: str = "gaussian",
    shape_scale: float = 1.0,
    angular_degree=None,
) -> dict[str, OnsiteSetup]:
    """One :class:`OnsiteSetup` per species, sharing the calculation's functional."""
    options = {} if angular_degree is None else {"angular_degree": angular_degree}
    return {
        symbol: OnsiteSetup(
            data,
            xc,
            make_shape(shape, data.rcut, scale=shape_scale),
            **options,
        )
        for symbol, data in datasets.items()
    }
