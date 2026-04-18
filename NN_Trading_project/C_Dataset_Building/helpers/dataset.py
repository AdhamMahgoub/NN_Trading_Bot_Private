"""
dataset.py
----------
PyTorch Dataset for the leakage-safe train/val/test stock splits.

Usage
-----
    from C_Dataset_Building.helpers.dataset import StockDatasetSafe

    train_ds = StockDatasetSafe(index, scaler, "train")
    val_ds   = StockDatasetSafe(index, scaler, "val")
    test_ds  = StockDatasetSafe(index, scaler, "test")
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def is_cache_valid(scaler_path, index_path) -> bool:
    """Return True only if the scaler, index, and every referenced .npz exist."""
    if not (Path(scaler_path).exists() and Path(index_path).exists()):
        return False
    import pickle
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    if not index:
        return False
    return all(Path(entry["cache_file"]).exists() for entry in index)


class StockDatasetSafe(Dataset):
    """Loads pre-cached .npz files and applies the fitted scaler.

    Parameters
    ----------
    index : list[dict]
        List returned by ``precompute_and_cache``.  Each dict must contain
        ``cache_file``, ``train_cut``, ``val_start``, ``val_end``,
        ``test_start``.
    scaler : sklearn.preprocessing.StandardScaler | None
        Fitted on train data only.  Pass None to skip scaling.
    split : {"train", "val", "test"}
    """

    def __init__(self, index: list, scaler, split: str):
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split '{split}'"
        self.scaler  = scaler
        self.split   = split
        self.samples = []

        for entry in index:
            cache_file = entry["cache_file"]
            if not Path(cache_file).exists():
                print(f"[Dataset] WARNING: missing cache file, skipping: {cache_file}")
                continue
            data = np.load(cache_file)
            X_full, y_full = data["X"], data["y"]

            train_cut  = int(entry["train_cut"])
            val_start  = int(entry["val_start"])
            val_end    = int(entry["val_end"])
            test_start = int(entry["test_start"])

            if split == "train":
                X, y = X_full[:train_cut],          y_full[:train_cut]
            elif split == "val":
                X, y = X_full[val_start:val_end],   y_full[val_start:val_end]
            else:
                X, y = X_full[test_start:],          y_full[test_start:]

            if len(X) == 0:
                continue

            if self.scaler is not None:
                X = self.scaler.transform(X).astype(np.float32, copy=False)

            self.samples.extend((X[i], y[i]) for i in range(len(X)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        X, y = self.samples[idx]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)
