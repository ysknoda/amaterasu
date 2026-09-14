"""HDF5 output: the result of a calculation, in the documented output units.

Every dataset carries a ``units`` attribute naming what it is in, and the file-level
attributes record the unit convention as a whole. Internally amaterasu works in Hartree
atomic units; the conversion happens here and nowhere else, so a reader never has to guess.

Layout
------
::

    /                       attributes: version, provenance, unit convention
    /structure              cell (A), positions (A), numbers, magmoms
    /basis                  ecut, ecut_rho (eV), grid shape, plane waves per k-point
    /kpoints                fractional coordinates, weights
    /scf                    energies (eV), eigenvalues (eV), occupations, Fermi level
    /density                electrons per A^3, shape (nspin, *grid)
    /bands                  path coordinates, eigenvalues (eV), labels  [solver: bands]

Wavefunctions are written only when ``output.save_wavefunctions`` is set: they are by far
the largest object in the calculation and are rarely wanted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..units import BOHR_IN_ANG, HARTREE_IN_EV, OUTPUT_UNITS
from .provenance import provenance

__all__ = ["read_bands", "read_scf", "write_bands", "write_scf"]


def _attrs(dataset, units: str, description: str) -> None:
    dataset.attrs["units"] = units
    dataset.attrs["description"] = description


def write_scf(
    path: str | Path,
    system,
    result,
    *,
    config=None,
    pseudopotential_paths: dict[str, str] | None = None,
    save_wavefunctions: bool = False,
) -> Path:
    """Write an SCF result to HDF5 and return the path."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in provenance(pseudopotential_paths).items():
            handle.attrs[key] = value
        for quantity, unit in OUTPUT_UNITS.items():
            handle.attrs[f"unit_{quantity}"] = unit
        handle.attrs["solver"] = "scf"
        handle.attrs["functional"] = system.xc.name
        handle.attrs["converged"] = bool(result.converged)
        if config is not None:
            from ..config import config_to_yaml

            handle.attrs["resolved_input"] = config_to_yaml(config)

        structure = handle.create_group("structure")
        _attrs(
            structure.create_dataset("cell", data=system.cell * BOHR_IN_ANG),
            "Angstrom",
            "lattice vectors as rows",
        )
        _attrs(
            structure.create_dataset("positions", data=system.positions * BOHR_IN_ANG),
            "Angstrom",
            "Cartesian atomic positions",
        )
        structure.create_dataset("numbers", data=np.asarray(system.atoms.numbers))
        structure.create_dataset(
            "symbols", data=np.asarray(system.symbols, dtype=h5py.string_dtype())
        )
        _attrs(
            structure.create_dataset("initial_magmoms", data=system.initial_magmoms),
            "Bohr magneton",
            "initial moments the SCF was started from",
        )

        basis = handle.create_group("basis")
        basis.attrs["ecut"] = system.ecut * HARTREE_IN_EV
        basis.attrs["ecut_rho"] = system.ecut_rho * HARTREE_IN_EV
        basis.attrs["units"] = "eV"
        basis.create_dataset("grid_shape", data=np.asarray(system.grid.shape))
        basis.create_dataset("plane_waves", data=np.asarray([len(s) for s in system.wave_spheres]))

        kpoints = handle.create_group("kpoints")
        _attrs(
            kpoints.create_dataset("points", data=system.kpoints.points),
            "fractional",
            "coordinates in the reciprocal cell",
        )
        kpoints.create_dataset("weights", data=system.kpoints.weights)
        kpoints.attrs["symmetry_reduced"] = bool(system.kpoints.symmetry_reduced)
        kpoints.attrs["n_full"] = int(system.kpoints.n_full)

        scf = handle.create_group("scf")
        scf.attrs["iterations"] = int(result.iterations)
        scf.attrs["seconds"] = float(result.seconds)
        scf.attrs["electron_count"] = float(result.electron_count)
        scf.attrs["magnetization"] = float(result.magnetization)
        energies = scf.create_group("energies")
        energies.attrs["units"] = "eV"
        for name in ("kinetic", "local", "hartree", "xc", "nonlocal_", "ewald", "entropy"):
            energies.attrs[name.rstrip("_")] = getattr(result.energies, name) * HARTREE_IN_EV
        energies.attrs["total"] = result.energies.total * HARTREE_IN_EV
        energies.attrs["free_energy"] = result.energies.free_energy * HARTREE_IN_EV
        energies.attrs["extrapolated"] = result.energies.extrapolated * HARTREE_IN_EV
        _attrs(
            scf.create_dataset("eigenvalues", data=result.eigenvalues * HARTREE_IN_EV),
            "eV",
            "(nspin, nkpoints, nbands)",
        )
        _attrs(
            scf.create_dataset("occupations", data=result.occupations),
            "dimensionless",
            "occupation of each state, between 0 and 1",
        )
        scf.attrs["fermi_level"] = result.fermi_level * HARTREE_IN_EV
        if result.gap is not None:
            scf.attrs["gap"] = result.gap * HARTREE_IN_EV
        history = np.asarray(result.history, dtype=float)
        if history.size:
            history[:, 0] *= HARTREE_IN_EV
            history[:, 2] *= HARTREE_IN_EV
            _attrs(
                scf.create_dataset("history", data=history),
                "eV, electrons per electron, eV",
                "per iteration: free energy, density residual, eigenvalue-sum residual",
            )

        _attrs(
            handle.create_dataset(
                "density", data=result.density / BOHR_IN_ANG**3, compression="gzip"
            ),
            "electrons/Angstrom^3",
            "(nspin, *grid_shape) on the real-space grid",
        )
        if save_wavefunctions:
            group = handle.create_group("wavefunctions")
            group.attrs["description"] = (
                "coefficients of exp(i(k+G).r)/sqrt(Omega); sum_G |c|^2 = 1"
            )
            for spin, per_spin in enumerate(result.wavefunctions):
                for ik, block in enumerate(per_spin):
                    group.create_dataset(f"spin{spin}/kpoint{ik}", data=block, compression="gzip")
                    group.create_dataset(
                        f"spin{spin}/miller{ik}", data=system.wave_spheres[ik].miller
                    )
    return path


def read_scf(path: str | Path) -> dict:
    """Read back the parts of an SCF result needed to restart or post-process it."""
    import h5py

    with h5py.File(Path(path), "r") as handle:
        result = {
            "attrs": dict(handle.attrs),
            "eigenvalues": handle["scf/eigenvalues"][()],
            "occupations": handle["scf/occupations"][()],
            "energies": dict(handle["scf/energies"].attrs),
            "density": handle["density"][()] * BOHR_IN_ANG**3,
            "kpoints": handle["kpoints/points"][()],
            "weights": handle["kpoints/weights"][()],
            "cell": handle["structure/cell"][()],
            "positions": handle["structure/positions"][()],
            "symbols": [
                s.decode() if isinstance(s, bytes) else s for s in handle["structure/symbols"][()]
            ],
        }
        if "wavefunctions" in handle:
            group = handle["wavefunctions"]
            result["wavefunctions"] = [
                [group[f"spin{spin}/kpoint{ik}"][()] for ik in range(len(result["kpoints"]))]
                for spin in range(len(group))
            ]
    return result


def write_bands(path: str | Path, structure, *, config=None) -> Path:
    """Write a band structure to HDF5 and return the path.

    The eigenvalues are stored as they were computed, and the Fermi level alongside them,
    rather than subtracted: a reader that wants them relative to the Fermi level can do
    that, and one that wants absolute values cannot undo it.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    band_path = structure.path
    with h5py.File(path, "w") as handle:
        for key, value in provenance(None).items():
            handle.attrs[key] = value
        for quantity, unit in OUTPUT_UNITS.items():
            handle.attrs[f"unit_{quantity}"] = unit
        handle.attrs["solver"] = "bands"
        handle.attrs["path"] = str(band_path.path)
        handle.attrs["restandardised_cell"] = bool(structure.restandardised)
        if config is not None:
            from ..config import config_to_yaml

            handle.attrs["resolved_input"] = config_to_yaml(config)

        group = handle.create_group("bands")
        distance, special_positions, special_labels = band_path.get_linear_kpoint_axis()
        _attrs(
            group.create_dataset("kpoints", data=np.asarray(band_path.kpts)),
            "fractional",
            "path coordinates in the reciprocal cell",
        )
        _attrs(
            group.create_dataset("distance", data=np.asarray(distance)),
            "1/Angstrom",
            "distance along the path, for the horizontal axis of a plot",
        )
        _attrs(
            group.create_dataset("eigenvalues", data=structure.eigenvalues * HARTREE_IN_EV),
            "eV",
            "eigenvalues, shape (nspin, nkpoints, nbands)",
        )
        _attrs(
            group.create_dataset("fermi_level", data=structure.fermi_level * HARTREE_IN_EV),
            "eV",
            "Fermi level from the self-consistent mesh, not from the path",
        )
        group.create_dataset("special_point_positions", data=np.asarray(special_positions))
        group.create_dataset(
            "special_point_labels", data=np.asarray(special_labels, dtype=h5py.string_dtype())
        )

        cell = np.asarray(structure.scf.system.cell) * BOHR_IN_ANG
        _attrs(
            handle.create_dataset("structure/cell", data=cell),
            "Angstrom",
            "lattice vectors as rows -- the cell the path is defined in",
        )
    return path


def read_bands(path: str | Path) -> dict:
    """Read back a band structure written by :func:`write_bands`."""
    import h5py

    with h5py.File(Path(path), "r") as handle:
        return {
            "attrs": dict(handle.attrs),
            "kpoints": handle["bands/kpoints"][()],
            "distance": handle["bands/distance"][()],
            "eigenvalues": handle["bands/eigenvalues"][()],
            "fermi_level": float(handle["bands/fermi_level"][()]),
            "special_point_positions": handle["bands/special_point_positions"][()],
            "special_point_labels": [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in handle["bands/special_point_labels"][()]
            ],
            "cell": handle["structure/cell"][()],
        }
