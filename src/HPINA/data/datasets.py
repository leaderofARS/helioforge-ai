"""
src/HPINA/data/datasets.py
──────────────────────────
Sliding Window Dataset Generators for TCN / HPINA

Provides two generators:

  WindowGenerator
      Slices the raw 2-channel flux streams (SoLEXS COUNTS + HEL1OS energy)
      into 3D sequence tensors of shape (N_windows, Channels=2, Window_Size).
      Used for raw-signal baseline experiments.

  MultivariateFeatureWindowGenerator
      Implements the Chapter 1 mathematical formulation exactly:
          x_t ∈ R^F   (F-dim feature vector at time t)
          X  ∈ R^(T×F) (observation matrix)
          W_t∈ R^(F×L) (sliding temporal window, stored channels-first for PyTorch)

      Accepts the engineered feature matrix (selected_features.csv → F=38
      or all_features.csv → F=79) and produces tensors of shape
      (N_windows, F, Window_Size).

Both generators share a common _BaseWindowGenerator that owns the
train/val/test split logic and single-pass observation loading used by
generate_all_scales(), eliminating duplication.

Output tensors are saved to the windows directory defined in data_paths.yaml.
Per-feature min/max scalers fitted on the train split are persisted as
<output_dir>/scaler_f{F}_w{L}.json to guarantee reproducible normalisation
at inference time.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.pipeline.ingestion.observation_loader import ObservationLoader
from src.utils.config import PATH_CFG

logger = logging.getLogger("helioforge.hpina.datasets")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SolarSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for pre-built TCN sequence window tensors.

    Parameters
    ----------
    sequences : torch.Tensor
        3D tensor of shape (N_samples, Channels, Window_Size).
    targets : torch.Tensor, optional
        Target flare intensity or class labels (N_samples,).
    """

    def __init__(
        self,
        sequences: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> None:
        self.sequences = sequences
        self.targets   = targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.targets is not None:
            return self.sequences[idx], self.targets[idx]
        return self.sequences[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Shared base — split logic & single-pass observation loading
# ─────────────────────────────────────────────────────────────────────────────

class _BaseWindowGenerator(ABC):
    """
    Abstract base class shared by WindowGenerator and MultivariateFeatureWindowGenerator.

    Owns:
    - observation-level train / val / test split (no data leakage)
    - single-pass observation loading for generate_all_scales()
    - .pt saving helpers
    """

    def __init__(
        self,
        window_size: int = 512,
        stride: int = 32,
        train_ratio: float = 0.70,
        val_ratio: float   = 0.15,
        test_ratio: float  = 0.15,
        output_dir: Optional[Path] = None,
    ) -> None:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        self.window_size  = window_size
        self.stride       = stride
        self.train_ratio  = train_ratio
        self.val_ratio    = val_ratio
        self.test_ratio   = test_ratio
        self.output_dir   = output_dir if output_dir is not None else PATH_CFG.windows.root

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_windows_from_obs(self, obs: dict) -> np.ndarray:
        """Convert a single loaded observation dict to a numpy window array."""

    # ------------------------------------------------------------------
    # Observation loading (called once per generate_all / generate_all_scales)
    # ------------------------------------------------------------------

    def _load_observations(self) -> List[dict]:
        logger.info("Loading processed observations …")
        loader = ObservationLoader(PATH_CFG.preprocessing.processed)
        observations = list(loader.load_all())
        if not observations:
            raise RuntimeError(
                f"No processed observations found under {PATH_CFG.preprocessing.processed}.\n"
                "Please run Stage 1 preprocessing (preprocess.py) first."
            )
        return observations

    def _split_observations(
        self, observations: List[dict]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        n       = len(observations)
        n_train = int(n * self.train_ratio)
        n_val   = int(n * self.val_ratio)
        train   = observations[:n_train]
        val     = observations[n_train : n_train + n_val]
        test    = observations[n_train + n_val :]
        logger.info(
            "Observation split — train: %d  val: %d  test: %d  (total: %d)",
            len(train), len(val), len(test), n,
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Window extraction for a list of observations
    # ------------------------------------------------------------------

    def _build_split_tensor(self, obs_list: List[dict], n_channels: int) -> np.ndarray:
        """Concatenate windows from a list of observations into one numpy array."""
        arrays = []
        for obs in obs_list:
            w = self._extract_windows_from_obs(obs)
            if len(w) > 0:
                arrays.append(w)
        if not arrays:
            return np.empty((0, n_channels, self.window_size), dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------

    def _save_tensors(
        self,
        train_arr: np.ndarray,
        val_arr: np.ndarray,
        test_arr: np.ndarray,
        suffix: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_t = torch.from_numpy(train_arr)
        val_t   = torch.from_numpy(val_arr)
        test_t  = torch.from_numpy(test_arr)

        train_path = self.output_dir / f"train{suffix}.pt"
        val_path   = self.output_dir / f"val{suffix}.pt"
        test_path  = self.output_dir / f"test{suffix}.pt"

        torch.save({"sequences": train_t}, train_path)
        torch.save({"sequences": val_t},   val_path)
        torch.save({"sequences": test_t},  test_path)

        self._print_summary(train_t, val_t, test_t, train_path, val_path, test_path)
        return train_t, val_t, test_t

    @staticmethod
    def _print_summary(
        train_t: torch.Tensor,
        val_t: torch.Tensor,
        test_t: torch.Tensor,
        train_path: Path,
        val_path: Path,
        test_path: Path,
    ) -> None:
        print("\n" + "=" * 60)
        print("  HELIO-FORGE HPINA  |  WINDOW GENERATION COMPLETE")
        print("=" * 60)
        print(f"  Train Tensor : {train_t.shape}  →  {train_path}")
        print(f"  Val Tensor   : {val_t.shape}    →  {val_path}")
        print(f"  Test Tensor  : {test_t.shape}   →  {test_path}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # generate_all  (single scale)
    # ------------------------------------------------------------------

    def generate_all(
        self,
        observations: Optional[List[dict]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Build train/val/test window tensors for this generator's window_size.

        Parameters
        ----------
        observations : list[dict], optional
            Pre-loaded observation list.  If None, observations are loaded
            from disk (useful when calling generate_all() standalone).
            Pass a pre-loaded list to avoid redundant disk I/O when calling
            generate_all_scales().
        """
        if observations is None:
            observations = self._load_observations()

        train_obs, val_obs, test_obs = self._split_observations(observations)

        n_channels = self._n_channels()
        train_arr  = self._build_split_tensor(train_obs, n_channels)
        val_arr    = self._build_split_tensor(val_obs,   n_channels)
        test_arr   = self._build_split_tensor(test_obs,  n_channels)

        suffix = f"_w{self.window_size}" if self.window_size != 512 else ""
        train_t, val_t, test_t = self._save_tensors(train_arr, val_arr, test_arr, suffix=suffix)

        return {"train": train_t, "val": val_t, "test": test_t}

    # ------------------------------------------------------------------
    # generate_all_scales  (multiple scales — loads observations ONCE)
    # ------------------------------------------------------------------

    def generate_all_scales(
        self,
        scales: Optional[List[Tuple[int, int]]] = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Generate window tensors for multiple (window_size, stride) pairs in a
        single pass — observations are loaded from disk exactly once.

        Parameters
        ----------
        scales : list of (window_size, stride) tuples, optional
            Defaults to [(256, 16), (512, 32), (1024, 64)].
        """
        if scales is None:
            scales = [(256, 16), (512, 32), (1024, 64)]

        # Load observations exactly once
        observations = self._load_observations()

        results = {}
        for win_size, stride in scales:
            logger.info("Generating w%d (stride=%d) …", win_size, stride)
            gen = self._clone(window_size=win_size, stride=stride)
            results[f"w{win_size}"] = gen.generate_all(observations=observations)

        return results

    @abstractmethod
    def _clone(self, window_size: int, stride: int) -> "_BaseWindowGenerator":
        """Return a new instance of the same class with different window_size/stride."""

    @abstractmethod
    def _n_channels(self) -> int:
        """Return the number of channels this generator produces."""


# ─────────────────────────────────────────────────────────────────────────────
# WindowGenerator — raw 2-channel flux windows
# ─────────────────────────────────────────────────────────────────────────────

class WindowGenerator(_BaseWindowGenerator):
    """
    Slices the raw 2-channel 1 Hz flux streams into sliding window tensors.

    Output shape:  (N_windows, 2, window_size)
    Channels:
      [0]  SoLEXS COUNTS   (Soft X-Ray, 1–8 Å)
      [1]  HEL1OS energy   (Hard X-Ray, >10 keV)
    """

    # ------------------------------------------------------------------
    # Core windowing — single observation
    # ------------------------------------------------------------------

    def _slice_two_channel(
        self, soft_signal: np.ndarray, hard_signal: np.ndarray
    ) -> np.ndarray:
        """
        Slice one observation into windows of shape (N, 2, window_size).
        """
        length = min(len(soft_signal), len(hard_signal))
        if length < self.window_size:
            return np.empty((0, 2, self.window_size), dtype=np.float32)

        soft_clean = np.nan_to_num(soft_signal[:length], nan=0.0, posinf=0.0, neginf=0.0)
        hard_clean = np.nan_to_num(hard_signal[:length], nan=0.0, posinf=0.0, neginf=0.0)

        def _minmax(arr: np.ndarray) -> np.ndarray:
            lo, hi  = arr.min(), arr.max()
            rng     = (hi - lo) if (hi - lo) > 1e-8 else 1.0
            return (arr - lo) / rng

        stacked = np.vstack([_minmax(soft_clean), _minmax(hard_clean)]).astype(np.float32)

        windows = [
            stacked[:, s : s + self.window_size]
            for s in range(0, length - self.window_size + 1, self.stride)
        ]
        valid = [w for w in windows if not np.isnan(w).any() and not np.isinf(w).any()]

        if not valid:
            return np.empty((0, 2, self.window_size), dtype=np.float32)
        return np.stack(valid, axis=0)

    # ------------------------------------------------------------------
    # _BaseWindowGenerator interface
    # ------------------------------------------------------------------

    def _extract_windows_from_obs(self, obs: dict) -> np.ndarray:
        return self._slice_two_channel(obs["soft_signal"], obs["hard_signal"])

    def _n_channels(self) -> int:
        return 2

    def _clone(self, window_size: int, stride: int) -> "WindowGenerator":
        return WindowGenerator(
            window_size=window_size,
            stride=stride,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            output_dir=self.output_dir,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MultivariateFeatureWindowGenerator — Chapter 1 formulation
# ─────────────────────────────────────────────────────────────────────────────

class MultivariateFeatureWindowGenerator(_BaseWindowGenerator):
    """
    Implements the Chapter 1 mathematical formulation exactly:

        x_t  ∈  R^F          instantaneous feature vector
        X    ∈  R^(T × F)    observation matrix (T timestamps, F features)
        W_t  ∈  R^(F × L)    sliding temporal window (channels-first for PyTorch)

    Accepts the pre-built engineered feature matrix loaded from a CSV file:
      - selected_features.csv  →  F = 38  (physics-informed selected features)
      - all_features.csv       →  F = 79  (full candidate feature space)

    Output tensor shape:  (N_windows, F, window_size)

    Normalisation
    -------------
    Min/max scalers are fitted on the TRAIN split only and persisted to
    <output_dir>/scaler_f{F}_w{window_size}.json so that val/test windows
    and runtime inference use exactly the same scaling parameters.

    Parameters
    ----------
    features_csv : Path
        Absolute path to the feature matrix CSV.  Must have a TIME column
        and F numeric feature columns.
    window_size : int
        Temporal window length L (default 512 timesteps = ~8.5 minutes at 1 Hz).
    stride : int
        Sliding window step (default 32).
    time_col : str
        Name of the time column in the CSV (default "TIME").
    """

    def __init__(
        self,
        features_csv: Path,
        window_size: int = 512,
        stride: int = 32,
        train_ratio: float = 0.70,
        val_ratio: float   = 0.15,
        test_ratio: float  = 0.15,
        output_dir: Optional[Path] = None,
        time_col: str = "TIME",
    ) -> None:
        super().__init__(
            window_size=window_size,
            stride=stride,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            output_dir=output_dir,
        )
        self.features_csv = Path(features_csv)
        self.time_col     = time_col

        # Filled during generate_all()
        self._train_min: Optional[np.ndarray] = None
        self._train_max: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Feature matrix loading
    # ------------------------------------------------------------------

    def _load_feature_dataframe(self) -> pd.DataFrame:
        """Load the feature CSV as a pandas DataFrame."""
        if not self.features_csv.exists():
            raise FileNotFoundError(
                f"Feature CSV not found: {self.features_csv}\n"
                "Run  python scripts/features.py  first to generate it."
            )
        return pd.read_csv(self.features_csv)

    # ------------------------------------------------------------------
    # Scaler persistence & window slicing helpers
    # ------------------------------------------------------------------

    def _scaler_path(self, F: int) -> Path:
        return self.output_dir / f"scaler_f{F}_w{self.window_size}.json"

    def _fit_save_scaler(self, train_matrix: np.ndarray) -> None:
        """Fit min/max on train_matrix (T_train, F) and persist to JSON."""
        self._train_min = np.nanmin(train_matrix, axis=0)
        self._train_max = np.nanmax(train_matrix, axis=0)
        F = train_matrix.shape[1]
        scaler_dict = {
            "feature_names": self._feature_names,
            "min": self._train_min.tolist(),
            "max": self._train_max.tolist(),
            "window_size": self.window_size,
            "n_features": F,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = self._scaler_path(F)
        with open(scaler_path, "w") as fh:
            json.dump(scaler_dict, fh, indent=2)
        logger.info("Scaler saved → %s", scaler_path)

    def _apply_scaler(self, matrix: np.ndarray) -> np.ndarray:
        """Apply train-fitted min/max normalisation to any split matrix."""
        rng = np.where(
            (self._train_max - self._train_min) > 1e-8,
            self._train_max - self._train_min,
            1.0,
        )
        return (matrix - self._train_min) / rng

    def _slice_normalised_matrix(self, norm_matrix: np.ndarray) -> np.ndarray:
        """Slice 2D matrix (T, F) into 3D window array (N_windows, F, window_size)."""
        T, F = norm_matrix.shape
        if T < self.window_size:
            return np.empty((0, F, self.window_size), dtype=np.float32)

        mat_t = norm_matrix.T.astype(np.float32)
        windows = [
            mat_t[:, s : s + self.window_size]
            for s in range(0, T - self.window_size + 1, self.stride)
        ]
        valid = [w for w in windows if not np.isnan(w).any() and not np.isinf(w).any()]
        if not valid:
            return np.empty((0, F, self.window_size), dtype=np.float32)
        return np.stack(valid, axis=0)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------


    def generate_all(
        self,
        observations: Optional[List[dict]] = None,  # unused — kept for API symmetry
    ) -> dict[str, torch.Tensor]:
        """
        Full pipeline:
          1. Load feature DataFrame from CSV.
          2. Perform Stratified Observation-Level Splitting by observation peak flare class.
          3. Fit scaler on train split only; persist to JSON.
          4. Normalise all three splits with train scaler.
          5. Slice each split into sliding windows.
          6. Save train.pt, val.pt, test.pt and return tensors.
        """
        df = self._load_feature_dataframe()
        meta_cols = {self.time_col, "observation_id"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df_feats = df[feature_cols].select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        self._feature_names = list(df_feats.columns)
        feature_cols = self._feature_names
        F = len(feature_cols)

        # Sanitize feature columns globally before splitting
        df[feature_cols] = np.nan_to_num(df[feature_cols].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

        # Stratified Observation-Level Split vs Fallback
        if "observation_id" in df.columns and df["observation_id"].nunique() > 1:
            logger.info("Performing Stratified Observation-Level Splitting across %d observations …", df["observation_id"].nunique())
            
            # Reconstruct peak flux per observation to determine flare class
            peak_col = "soft_mean" if "soft_mean" in df.columns else feature_cols[0]
            obs_summary = df.groupby("observation_id")[peak_col].max().reset_index()

            def _get_flare_class(max_val: float) -> str:
                if max_val < 100: return "Quiet"
                elif max_val < 500: return "B"
                elif max_val < 2000: return "C"
                elif max_val < 8000: return "M"
                else: return "X"

            obs_summary["flare_class"] = obs_summary[peak_col].apply(_get_flare_class)

            train_obs, val_obs, test_obs = [], [], []
            rng = np.random.RandomState(42)

            for cls_name, group in obs_summary.groupby("flare_class"):
                obs_list = group["observation_id"].tolist()
                rng.shuffle(obs_list)
                n = len(obs_list)
                if n >= 3:
                    n_tr = max(1, int(round(n * self.train_ratio)))
                    n_va = 1
                    n_te = n - n_tr - n_va
                    if n_te <= 0:
                        n_tr -= 1
                        n_te = 1
                elif n == 2:
                    n_tr, n_va, n_te = 1, 1, 0
                else:
                    n_tr, n_va, n_te = 1, 0, 0

                train_obs.extend(obs_list[:n_tr])
                val_obs.extend(obs_list[n_tr : n_tr + n_va])
                test_obs.extend(obs_list[n_tr + n_va :])


            logger.info(
                "Stratified Observation Split — Train: %d obs, Val: %d obs, Test: %d obs",
                len(train_obs), len(val_obs), len(test_obs),
            )

            train_df = df[df["observation_id"].isin(train_obs)]
            val_df   = df[df["observation_id"].isin(val_obs)]
            test_df  = df[df["observation_id"].isin(test_obs)]

            train_mat = train_df[feature_cols].to_numpy(dtype=np.float64)
            val_mat   = val_df[feature_cols].to_numpy(dtype=np.float64)
            test_mat  = test_df[feature_cols].to_numpy(dtype=np.float64)
        else:
            # Fallback for single-observation or missing observation_id
            matrix = df[feature_cols].to_numpy(dtype=np.float64)
            T = len(matrix)
            n_train = int(T * self.train_ratio)
            n_val   = int(T * self.val_ratio)

            train_mat = matrix[:n_train]
            val_mat   = matrix[n_train : n_train + n_val]
            test_mat  = matrix[n_train + n_val :]

        logger.info(
            "Feature matrix split — train: %d rows  val: %d rows  test: %d rows  (F=%d)",
            len(train_mat), len(val_mat), len(test_mat), F,
        )

        # Fit + persist scaler on train only
        self._fit_save_scaler(train_mat)

        # Normalise all splits with train scaler
        train_norm = self._apply_scaler(train_mat)
        val_norm   = self._apply_scaler(val_mat)
        test_norm  = self._apply_scaler(test_mat)

        # Slice into windows
        train_arr = self._slice_normalised_matrix(train_norm)
        val_arr   = self._slice_normalised_matrix(val_norm)
        test_arr  = self._slice_normalised_matrix(test_norm)

        # File name suffix encodes both F and L for clarity
        suffix = f"_feat{F}_w{self.window_size}"
        train_t, val_t, test_t = self._save_tensors(train_arr, val_arr, test_arr, suffix=suffix)

        print(f"  Channels (F) : {F} features  ({self.features_csv.name})")
        print(f"  Window Size  : {self.window_size} timesteps")
        print(f"  Stride       : {self.stride} timesteps")
        print(f"  Scaler       : {self._scaler_path(F)}")
        print("=" * 60)

        return {"train": train_t, "val": val_t, "test": test_t}


    # ------------------------------------------------------------------
    # _BaseWindowGenerator interface (stubs — not used for feature mode)
    # ------------------------------------------------------------------

    def _extract_windows_from_obs(self, obs: dict) -> np.ndarray:
        """Not used — feature mode slices the CSV matrix directly."""
        raise NotImplementedError(
            "MultivariateFeatureWindowGenerator slices the feature CSV matrix directly. "
            "Call generate_all() instead of using the observation-level API."
        )

    def _n_channels(self) -> int:
        # Resolved at runtime from the CSV; placeholder returns 0
        return 0

    def _clone(self, window_size: int, stride: int) -> "MultivariateFeatureWindowGenerator":
        return MultivariateFeatureWindowGenerator(
            features_csv=self.features_csv,
            window_size=window_size,
            stride=stride,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            output_dir=self.output_dir,
            time_col=self.time_col,
        )

    # ------------------------------------------------------------------
    # Multi-scale convenience  (overrides base to skip observation loading)
    # ------------------------------------------------------------------

    def generate_all_scales(
        self,
        scales: Optional[List[Tuple[int, int]]] = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Generate feature window tensors for multiple window sizes.
        The CSV is loaded once; each scale gets its own train-fitted scaler.
        """
        if scales is None:
            scales = [(256, 16), (512, 32), (1024, 64)]

        results = {}
        for win_size, stride in scales:
            logger.info("Generating feature windows w%d (stride=%d) …", win_size, stride)
            gen = self._clone(window_size=win_size, stride=stride)
            results[f"w{win_size}"] = gen.generate_all()

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load_scaler
# ─────────────────────────────────────────────────────────────────────────────

def load_scaler(scaler_json: Path) -> dict:
    """
    Load a persisted scaler produced by MultivariateFeatureWindowGenerator.

    Returns a dict with keys: feature_names, min, max, window_size, n_features.
    """
    with open(scaler_json, "r") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running raw 2-channel WindowGenerator (window=512, stride=32) …")
    generator = WindowGenerator(window_size=512, stride=32)
    generator.generate_all()
