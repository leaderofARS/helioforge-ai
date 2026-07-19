"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
feature_fusion.py

Combine all feature extraction modules into a
single machine-learning ready feature dictionary.
==========================================================
"""

from __future__ import annotations

import numpy as np

from src.features.correlation_features import (
    CorrelationFeatureExtractor,
)
from src.features.entropy_features import (
    EntropyFeatureExtractor,
)
from src.features.frequency_features import (
    FrequencyFeatureExtractor,
)
from src.features.hard_features import (
    HardFeatureExtractor,
)
from src.features.soft_features import (
    SoftFeatureExtractor,
)
from src.features.temporal_features import (
    TemporalFeatureExtractor,
)
from src.features.wavelet_features import (
    WaveletFeatureExtractor,
)


class FeatureFusion:
    """
    Combines all feature extraction modules into a
    single feature dictionary.
    """

    def __init__(self) -> None:

        self.soft = SoftFeatureExtractor()

        self.hard = HardFeatureExtractor()

        self.temporal = TemporalFeatureExtractor()

        self.frequency = FrequencyFeatureExtractor()

        self.wavelet = WaveletFeatureExtractor()

        self.entropy = EntropyFeatureExtractor()

        self.correlation = CorrelationFeatureExtractor()

    def extract(
        self,
        soft_signal: np.ndarray,
        hard_signal: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, float]:
        """
        Extract every feature from both signals.

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
            Complete engineered feature dictionary.
        """

        ##################################################
        # INPUT VALIDATION
        ##################################################

        if (
            len(soft_signal)
            != len(hard_signal)
            or len(soft_signal)
            != len(timestamps)
        ):
            raise ValueError(
                "Soft signal, hard signal and timestamps "
                "must have equal length."
            )

        features: dict[str, float] = {}

        ##################################################
        # SOFT FEATURES
        ##################################################

        features.update(
            self.soft.extract(
                soft_signal,
                timestamps,
            )
        )

        ##################################################
        # HARD FEATURES
        ##################################################

        features.update(
            self.hard.extract(
                hard_signal,
                timestamps,
            )
        )

        ##################################################
        # TEMPORAL FEATURES
        ##################################################

        features.update(
            self.temporal.extract(
                soft_signal,
                timestamps,
            )
        )

        ##################################################
        # FREQUENCY FEATURES
        ##################################################

        features.update(
            self.frequency.extract(
                soft_signal,
                timestamps,
            )
        )

        ##################################################
        # WAVELET FEATURES
        ##################################################

        features.update(
            self.wavelet.extract(
                soft_signal,
                timestamps,
            )
        )

        ##################################################
        # ENTROPY FEATURES
        ##################################################

        features.update(
            self.entropy.extract(
                soft_signal,
                timestamps,
            )
        )

        ##################################################
        # CORRELATION FEATURES
        ##################################################

        features.update(
            self.correlation.extract(
                soft_signal,
                hard_signal,
                timestamps,
            )
        )

        ##################################################
        # RETURN
        ##################################################

        return features