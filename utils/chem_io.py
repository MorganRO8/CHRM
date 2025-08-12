from __future__ import annotations

from typing import Dict

import torch

from dataset.common import ChemDatasetMetadata


def save_xyz(
    preds: Dict[str, torch.Tensor],
    metadata: ChemDatasetMetadata,
    filename: str,
) -> None:
    """Write predictions to an XYZ file.

    Parameters
    ----------
    preds : Dict[str, torch.Tensor]
        Dictionary containing ``atom_types`` and ``positions`` and optionally
        ``energy``.
    metadata : ChemDatasetMetadata
        Dataset metadata used to determine padding value.
    filename : str
        Destination XYZ filename.
    """
    from ase.data import chemical_symbols

    atom_types = preds["atom_types"].cpu().numpy()
    positions = preds["positions"].cpu().numpy()
    energy = preds.get("energy")

    with open(filename, "w") as fh:
        for i in range(atom_types.shape[0]):
            mask = atom_types[i] != metadata.pad_atom_type
            numbers = atom_types[i][mask].astype(int)
            coords = positions[i][mask]

            fh.write(f"{numbers.size}\n")
            if energy is not None:
                fh.write(f"Energy={float(energy[i]):.6f}\n")
            else:
                fh.write("\n")

            for n, (x, y, z) in zip(numbers, coords):
                symbol = (
                    chemical_symbols[int(n)]
                    if int(n) < len(chemical_symbols)
                    else str(int(n))
                )
                fh.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


def save_ase_trajectory(
    preds: Dict[str, torch.Tensor],
    metadata: ChemDatasetMetadata,
    filename: str,
) -> None:
    """Write predictions to an ASE trajectory file.

    Parameters
    ----------
    preds : Dict[str, torch.Tensor]
        Dictionary containing ``atom_types`` and ``positions`` and optionally
        ``energy`` and ``forces``.
    metadata : ChemDatasetMetadata
        Dataset metadata used to determine padding value.
    filename : str
        Destination trajectory filename.
    """
    try:
        from ase import Atoms  # type: ignore
        from ase.io.trajectory import Trajectory  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ASE is required to write trajectory files"
        ) from exc

    atom_types = preds["atom_types"].cpu().numpy()
    positions = preds["positions"].cpu().numpy()
    energy = preds.get("energy")
    forces = preds.get("forces")

    with Trajectory(filename, "w") as traj:
        for i in range(atom_types.shape[0]):
            mask = atom_types[i] != metadata.pad_atom_type
            numbers = atom_types[i][mask].astype(int)
            coords = positions[i][mask]

            atoms = Atoms(numbers=numbers, positions=coords)
            if energy is not None:
                atoms.info["energy"] = float(energy[i])
            if forces is not None:
                atoms.arrays["forces"] = forces[i][mask].cpu().numpy()
            traj.write(atoms)


__all__ = ["save_xyz", "save_ase_trajectory"]
