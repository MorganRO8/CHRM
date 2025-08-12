import os
import json
from typing import Dict

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset.common import ChemDatasetMetadata


class ChemDatasetConfig(pydantic.BaseModel):
    """Configuration for :class:`ChemDataset`."""

    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool

    epochs_per_iter: int

    rank: int
    num_replicas: int


class ChemDataset(IterableDataset):
    """Iterable dataset for molecular data.

    The dataset is expected to be stored on disk in ``npy`` format with one
    directory per split (``train``, ``val`` or ``test``).  Each directory
    contains arrays ``{set_name}__atom_types.npy``, ``{set_name}__positions.npy``
    and ``{set_name}__energy.npy``.  If forces are available an additional
    ``{set_name}__forces.npy`` is stored.  A ``dataset.json`` file stores
    :class:`ChemDatasetMetadata` describing the dataset.
    """

    def __init__(self, config: ChemDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()

        assert (
            self.config.global_batch_size % self.config.num_replicas == 0
        ), f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        self._data: Dict[str, Dict[str, np.ndarray]] | None = None
        self._iters = 0

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _load_metadata(self) -> ChemDatasetMetadata:
        with open(os.path.join(self.config.dataset_path, self.split, "dataset.json"), "r") as f:
            return ChemDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "atom_types": "r",
            "positions": "r",
            "energy": "r",
        }
        if self.metadata.has_forces:
            field_mmap_modes["forces"] = "r"

        self._data = {}
        for set_name in self.metadata.sets:
            self._data[set_name] = {
                field_name: np.load(
                    os.path.join(
                        self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"
                    ),
                    mmap_mode=mmap_mode,
                )
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    # ------------------------------------------------------------------
    # Batching helpers
    # ------------------------------------------------------------------
    def _collate_batch(self, batch: Dict[str, np.ndarray]):
        if batch["atom_types"].shape[0] < self.local_batch_size:
            pad_size = self.local_batch_size - batch["atom_types"].shape[0]
            pad_values = {
                "atom_types": self.metadata.pad_atom_type,
                "positions": 0.0,
                "energy": 0.0,
            }
            if self.metadata.has_forces:
                pad_values["forces"] = 0.0

            for k, v in batch.items():
                pad_shape = ((0, pad_size),) + ((0, 0),) * (v.ndim - 1)
                batch[k] = np.pad(v, pad_shape, constant_values=pad_values[k])

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Iterators
    # ------------------------------------------------------------------
    def _iter_test(self):
        for set_name, dataset in self._data.items():  # type: ignore[union-attr]
            total_examples = len(dataset["atom_types"])

            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.config.global_batch_size)

                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(
                    start_index + (self.config.rank + 1) * self.local_batch_size,
                    end_index,
                )

                batch = self._collate_batch(
                    {
                        field: arr[local_start:local_end]
                        for field, arr in dataset.items()
                    }
                )

                yield set_name, batch, end_index - start_index

                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore[union-attr]
            self._iters += 1
            rng = np.random.Generator(
                np.random.Philox(seed=self.config.seed + self._iters)
            )

            total_examples = len(dataset["atom_types"])
            order = np.concatenate(
                [rng.permutation(total_examples) for _ in range(self.config.epochs_per_iter)]
            )
            start_index = 0

            while start_index < order.size:
                batch_indices = order[
                    start_index : start_index + self.config.global_batch_size
                ]
                start_index += self.config.global_batch_size

                if batch_indices.size < self.config.global_batch_size:
                    break

                local_indices = batch_indices[
                    self.config.rank * self.local_batch_size : (self.config.rank + 1)
                    * self.local_batch_size
                ]

                batch = self._collate_batch(
                    {field: arr[local_indices] for field, arr in dataset.items()}
                )

                yield set_name, batch, batch_indices.size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __iter__(self):
        worker_info = get_worker_info()
        assert (
            worker_info is None or worker_info.num_workers == 1
        ), "Multithreaded data loading is not currently supported."

        self._lazy_load_dataset()

        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


# ----------------------------------------------------------------------
# Dataset builders
# ----------------------------------------------------------------------
def build_qm9_dataset(output_dir: str, split_frac=(0.8, 0.1, 0.1), seed: int = 0):
    """Convert the QM9 dataset into the format expected by :class:`ChemDataset`.

    The QM9 dataset is downloaded using :mod:`torch_geometric` if necessary and
    then converted into ``npy`` arrays.  ``output_dir`` will contain ``train``,
    ``val`` and ``test`` splits each with a single set named ``all``.

    Parameters
    ----------
    output_dir: str
        Destination directory where the converted dataset will be written.
    split_frac: tuple
        Fractions for train/val/test splits.  Must sum to 1.0.
    seed: int
        Random seed used for shuffling before splitting.
    """

    try:
        from torch_geometric.datasets import QM9
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "torch_geometric is required to build the QM9 dataset"
        ) from exc

    dataset = QM9(root=output_dir)
    num_samples = len(dataset)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    n_train = int(split_frac[0] * num_samples)
    n_val = int(split_frac[1] * num_samples)

    split_indices = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }

    max_atoms = max(data.z.numel() for data in dataset)
    atom_types_set = sorted({int(z) for data in dataset for z in data.z.tolist()})
    pad_atom_type = 0
    atom_type_map = {z: i + 1 for i, z in enumerate(atom_types_set)}  # 0 is pad
    num_atom_types = len(atom_type_map) + 1
    pos_dim = dataset[0].pos.size(-1)

    for split, indices in split_indices.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        n = len(indices)
        atom_types = np.full((n, max_atoms), pad_atom_type, dtype=np.int16)
        positions = np.zeros((n, max_atoms, pos_dim), dtype=np.float32)
        energy = np.zeros((n,), dtype=np.float32)

        for i, idx in enumerate(indices):
            data = dataset[int(idx)]
            n_atoms = data.z.numel()
            atom_types[i, :n_atoms] = [atom_type_map[int(z)] for z in data.z.tolist()]
            positions[i, :n_atoms] = data.pos.numpy()
            energy[i] = float(data.y[0])  # use first target as energy

        np.save(os.path.join(split_dir, "all__atom_types.npy"), atom_types)
        np.save(os.path.join(split_dir, "all__positions.npy"), positions)
        np.save(os.path.join(split_dir, "all__energy.npy"), energy)

        metadata = ChemDatasetMetadata(
            pad_atom_type=pad_atom_type,
            num_atom_types=num_atom_types,
            max_atoms=max_atoms,
            pos_dim=pos_dim,
            has_forces=False,
            sets=["all"],
        )
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.dict(), f)


__all__ = ["ChemDataset", "ChemDatasetConfig", "build_qm9_dataset"]

