"""Data utilities for LDH-Net.

Expected raw array shape: [T, N, F]
Each sample becomes:
  X: [L, N, F]
  Y: [H, N, Fy]
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class StandardScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        # Fit per feature F over time and entities.
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std = x.std(axis=(0, 1), keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_target(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        return x * self.std[..., :target_dim] + self.mean[..., :target_dim]


class SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, input_len: int, horizon: int, target_dim: int = 1):
        if data.ndim != 3:
            raise ValueError(f"data must have shape [T,N,F], got {data.shape}")
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.input_len = input_len
        self.horizon = horizon
        self.target_dim = target_dim
        self.count = len(data) - input_len - horizon + 1
        if self.count <= 0:
            raise ValueError("Time series is shorter than input_len + horizon")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.horizon, :, : self.target_dim]
        return x, y


def chronological_split(data: np.ndarray, train_ratio=0.70, val_ratio=0.15):
    t = len(data)
    i1 = int(t * train_ratio)
    i2 = int(t * (train_ratio + val_ratio))
    return data[:i1], data[i1:i2], data[i2:]
