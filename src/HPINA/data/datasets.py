"""
src/HPINA/data/datasets.py
──────────────────────────
Sliding Window Dataset Generator for TCN / HPINA

Reads processed high-cadence time-series observations (SoLEXS soft X-rays & HEL1OS hard X-rays)
from /opt/helioforge-ai/data/preprocessing/processed/ and generates 3D sequence tensors:

    Tensor Shape : (N_windows, Channels=2, Window_Size=512)
    Channels     : [0] SoLEXS COUNTS (Soft X-Ray), [1] HEL1OS energy (Hard X-Ray)

Saves PyTorch tensors to /opt/helioforge-ai/data/windows/:
    - train.pt
    - val.pt
    - test.pt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.pipeline.ingestion.observation_loader import ObservationLoader
from src.utils.config import PATH_CFG

logger = logging.getLogger("helioforge.hpina.datasets")


class SolarSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for TCN sequence windows.
    """

    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """
        Parameters
        ----------
        sequences : torch.Tensor
            3D tensor of shape (N_samples, Channels, Window_Size)
        targets : torch.Tensor, optional
            Target flare intensity or class labels (N_samples,)
        """
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if self.targets is not None:
            return self.sequences[idx], self.targets[idx]
        return self.sequences[idx]


class WindowGenerator:
    """
    Slices multi-instrument time-series observations into sliding window tensors.
    """

    def __init__(
        self,
        window_size: int = 512,
        stride: int = 32,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        output_dir: Path | None = None,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.output_dir = output_dir if output_dir is not None else PATH_CFG.windows.root

    def create_windows_from_signal(
        self, soft_signal: np.ndarray, hard_signal: np.ndarray
    ) -> np.ndarray:
        """
        Slice a single observation into (N_windows, 2, window_size) numpy array.
        """
        length = min(len(soft_signal), len(hard_signal))
        if length < self.window_size:
            return np.empty((0, 2, self.window_size), dtype=np.float32)

        # Min-Max Normalize signals per observation to [0, 1] range safely
        s_min, s_max = np.min(soft_signal[:length]), np.max(soft_signal[:length])
        h_min, h_max = np.min(hard_signal[:length]), np.max(hard_signal[:length])

        s_norm = (soft_signal[:length] - s_min) / (s_max - s_min + 1e-8)
        h_norm = (hard_signal[:length] - h_min) / (h_max - h_min + 1e-8)

        # Stack into (2, length) matrix
        stacked = np.vstack([s_norm, h_norm]).astype(np.float32)

        windows = []
        for start in range(0, length - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = stacked[:, start:end]
            windows.append(window)

        if not windows:
            return np.empty((0, 2, self.window_size), dtype=np.float32)

        return np.stack(windows, axis=0)

    def generate_all(self) -> dict[str, torch.Tensor]:
        """
        Process all processed observations, split by observation to avoid data leakage,
        and save train.pt, val.pt, and test.pt.
        """
        logger.info("Loading processed observations...")
        loader = ObservationLoader(PATH_CFG.preprocessing.processed)
        all_observations = list(loader.load_all())

        if not all_observations:
            raise RuntimeError(
                f"No processed observations found under {PATH_CFG.preprocessing.processed}.\n"
                "Please run Stage 1 preprocessing (preprocess.py) first."
            )

        n_obs = len(all_observations)
        n_train = int(n_obs * self.train_ratio)
        n_val = int(n_obs * self.val_ratio)

        train_obs = all_observations[:n_train]
        val_obs = all_observations[n_train : n_train + n_val]
        test_obs = all_observations[n_train + n_val :]

        logger.info(
            "Observation split: %d train, %d val, %d test (Total: %d)",
            len(train_obs),
            len(val_obs),
            len(test_obs),
            n_obs,
        )

        def extract_split_windows(obs_list: list[dict]) -> torch.Tensor:
            split_windows = []
            for obs in obs_list:
                w = self.create_windows_from_signal(obs["soft_signal"], obs["hard_signal"])
                if len(w) > 0:
                    split_windows.append(w)
            if not split_windows:
                return torch.empty((0, 2, self.window_size), dtype=torch.float32)
            concatenated = np.concatenate(split_windows, axis=0)
            return torch.from_numpy(concatenated)

        train_tensor = extract_split_windows(train_obs)
        val_tensor = extract_split_windows(val_obs)
        test_tensor = extract_split_windows(test_obs)

        # Save to destination directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = PATH_CFG.windows.train
        val_path = PATH_CFG.windows.val
        test_path = PATH_CFG.windows.test

        torch.save({"sequences": train_tensor}, train_path)
        torch.save({"sequences": val_tensor}, val_path)
        torch.save({"sequences": test_tensor}, test_path)

        print("\n" + "=" * 60)
        print("  HELIO-FORGE HPINA  |  WINDOW GENERATION COMPLETE")
        print("=" * 60)
        print(f"  Window Size  : {self.window_size} timesteps")
        print(f"  Stride       : {self.stride} timesteps")
        print(f"  Train Tensor : {train_tensor.shape} → {train_path}")
        print(f"  Val Tensor   : {val_tensor.shape}   → {val_path}")
        print(f"  Test Tensor  : {test_tensor.shape}  → {test_path}")
        print("=" * 60)

        return {
            "train": train_tensor,
            "val": val_tensor,
            "test": test_tensor,
        }


if __name__ == "__main__":
    generator = WindowGenerator(window_size=512, stride=32)
    generator.generate_all()
