"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
entropy_features.py

Extract entropy-based features from
astronomical time-series observations.
==========================================================
"""

from __future__ import annotations

import numpy as np

from src.utils.feature_utils import (
    approximate_entropy,
    energy_entropy,
    histogram_entropy,
    remove_invalid,
    shannon_entropy,
)

##################################################
# CONFIGURATION
##################################################

MAX_ENTROPY_SAMPLES = 2048

class EntropyFeatureExtractor:
    """
    Extract entropy-based features
    from astronomical time-series.
    """

    def extract(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, float]:

        ##################################################
        # CLEAN DATA
        ##################################################

        signal, timestamps = remove_invalid(
            signal,
            timestamps,
        )
        
        ##################################################
        # DOWNSAMPLE FOR EXPENSIVE ENTROPY FEATURES
        ##################################################

        entropy_signal = signal

        if entropy_signal.size > MAX_ENTROPY_SAMPLES:

            indices = np.linspace(
                0,
                entropy_signal.size - 1,
                MAX_ENTROPY_SAMPLES,
                dtype=int,
            )

            entropy_signal = entropy_signal[indices]

        ##################################################
        # VALIDATION
        ##################################################

        if signal.size == 0:
            raise ValueError(
                "Signal is empty."
            )

        if signal.size != timestamps.size:
            raise ValueError(
                "Signal and timestamps must have equal length."
            )

        features: dict[str, float] = {}

        ##################################################
        # SHANNON ENTROPY
        ##################################################

        features["shannon_entropy"] = float(
            shannon_entropy(signal)
        )

        ##################################################
        # ENERGY ENTROPY
        ##################################################

        features["energy_entropy"] = float(
            energy_entropy(signal)
        )
        
        ##################################################
        # HISTOGRAM ENTROPY
        ##################################################

        features["histogram_entropy"] = float(
            histogram_entropy(signal)
        )

        ##################################################
        # APPROXIMATE ENTROPY
        ##################################################

        features["approximate_entropy"] = float(
            approximate_entropy(entropy_signal)
        )

        ##################################################
        # ENTROPY RATIO
        ##################################################

        histogram = features["histogram_entropy"]

        if histogram > 0:

            features["entropy_ratio"] = float(
                features["shannon_entropy"]
                / histogram
            )

        else:

            features["entropy_ratio"] = 0.0

        ##################################################
        # RETURN FEATURES
        ##################################################

        return features