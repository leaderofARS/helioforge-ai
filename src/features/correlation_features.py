"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
correlation_features.py

Extract correlation-based features between
SoLEXS (Soft X-ray) and HEL1OS (Hard X-ray).
==========================================================
"""

from __future__ import annotations

import numpy as np

from src.utils.feature_utils import (
    cross_correlation,
    lag_at_max_correlation,
    max_cross_correlation,
    pearson_correlation,
    remove_invalid,
    rms_difference,
    spearman_correlation,
)


class CorrelationFeatureExtractor:
    """
    Extract correlation features between
    two synchronized astronomical signals.
    """

    def extract(
        self,
        soft_signal: np.ndarray,
        hard_signal: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, float]:

        ##################################################
        # CLEAN DATA
        ##################################################

        mask = (
            np.isfinite(soft_signal)
            & np.isfinite(hard_signal)
            & np.isfinite(timestamps)
        )       

        soft_signal = soft_signal[mask]
        hard_signal = hard_signal[mask]
        timestamps = timestamps[mask]

        ##################################################
        # VALIDATION
        ##################################################

        if (
            soft_signal.size
            != hard_signal.size
        ):
            raise ValueError(
                "Signals must have equal length."
            )

        if soft_signal.size == 0:
            raise ValueError(
                "Signals are empty."
            )

        features: dict[str, float] = {}

        ##################################################
        # PEARSON
        ##################################################

        features["pearson_correlation"] = float(
            pearson_correlation(
                soft_signal,
                hard_signal,
            )
        )

        ##################################################
        # SPEARMAN
        ##################################################

        features["spearman_correlation"] = float(
            spearman_correlation(
                soft_signal,
                hard_signal,
            )
        )
        
        ##################################################
        # CROSS CORRELATION
        ##################################################

        corr = cross_correlation(
            soft_signal,
            hard_signal,
        )

        features["maximum_cross_correlation"] = float(
            max_cross_correlation(
                corr
            )
        )

        features["lag_at_maximum_correlation"] = float(
            lag_at_max_correlation(
                corr,
                soft_signal.size,
            )
        )
        
        if soft_signal.size < 2:
            raise ValueError(
                "Signals must contain at least two samples."
            )

        ##################################################
        # COVARIANCE
        ##################################################

        covariance = np.cov(
            soft_signal,
            hard_signal,
        )

        features["covariance"] = float(
            covariance[0, 1]
        )

        ##################################################
        # DIFFERENCE FEATURES
        ##################################################

        difference = (
            soft_signal
            - hard_signal
        )

        features["mean_difference"] = float(
            np.mean(
                difference
            )
        )

        features["std_difference"] = float(
            np.std(
                difference
            )
        )

        ##################################################
        # RMS DIFFERENCE
        ##################################################

        features["rms_difference"] = float(
            rms_difference(
                soft_signal,
                hard_signal,
            )
        )

        ##################################################
        # RETURN FEATURES
        ##################################################

        return features