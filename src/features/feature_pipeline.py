"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
feature_pipeline.py

High-level interface for extracting all engineered
features from synchronized solar observations.
==========================================================
"""

from __future__ import annotations

import numpy as np

from src.features.feature_fusion import (
    FeatureFusion,
)


class FeaturePipeline:
    """
    High-level feature extraction pipeline.

    This class serves as the single entry point for
    feature engineering. It automatically invokes
    every feature extraction module and returns one
    unified feature dictionary.
    """

    def __init__(self) -> None:

        self.fusion = FeatureFusion()

    def run(
        self,
        soft_signal: np.ndarray,
        hard_signal: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, float]:
        """
        Extract all engineered features.

        Parameters
        ----------
        soft_signal : np.ndarray
            SoLEXS soft X-ray signal.

        hard_signal : np.ndarray
            HEL1OS hard X-ray signal.

        timestamps : np.ndarray
            Observation timestamps.

        Returns
        -------
        dict[str, float]
            Complete feature vector.
        """

        return self.fusion.extract(
            soft_signal,
            hard_signal,
            timestamps,
        )

    def __call__(
        self,
        soft_signal: np.ndarray,
        hard_signal: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, float]:
        """
        Allow the pipeline object to be called directly.
        """

        return self.run(
            soft_signal,
            hard_signal,
            timestamps,
        )