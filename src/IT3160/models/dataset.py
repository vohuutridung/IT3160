"""PyTorch Dataset and DataLoader for HDF5 chess tensor records."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ChessTensorDataset(Dataset):
    """Memory-efficient HDF5 dataset for chess policy training.

    File handles are opened lazily upon the first request inside each worker
    process to guarantee thread/process safety in multi-worker DataLoaders.
    """

    def __init__(self, h5_path: str | Path, limit: Optional[int] = None):
        self.h5_path = str(Path(h5_path).resolve())
        if not Path(self.h5_path).is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            total_rows = f["y"].shape[0]

        if limit is not None and limit > 0:
            self._length = min(total_rows, limit)
        else:
            self._length = total_rows

        self._file: Optional[h5py.File] = None
        self._x = None
        self._y = None
        self._legal_mask = None
        self._phase = None

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", swmr=True)
            self._x = self._file["x"]
            self._y = self._file["y"]
            self._legal_mask = self._file["legal_mask"]
            self._phase = self._file["phase"]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset length {self._length}")

        self._ensure_open()

        x_raw = self._x[idx]
        y_raw = self._y[idx]
        mask_raw = self._legal_mask[idx]
        phase_raw = self._phase[idx]

        # Convert to PyTorch tensors
        # x: uint8 [17, 8, 8] -> float32
        x = torch.from_numpy(x_raw).float()
        # y: int64 scalar
        y = torch.tensor(int(y_raw), dtype=torch.long)
        # legal_mask: bool [4672]
        legal_mask = torch.from_numpy(mask_raw).bool()
        # phase: uint8 scalar
        phase = torch.tensor(int(phase_raw), dtype=torch.long)

        return x, y, legal_mask, phase

    def __del__(self) -> None:
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            except Exception:
                pass


def create_dataloader(
    h5_path: str | Path,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    limit: Optional[int] = None,
) -> DataLoader:
    """Create a high-performance DataLoader from an HDF5 tensor file."""
    dataset = ChessTensorDataset(h5_path, limit=limit)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
