"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
temporal_features.py

Extract temporal features from
time-series observations.
==========================================================
"""

from __future__ import annotations

import numpy as np

from src.utils.feature_utils import (
    autocorrelation,
    compute_gradient,
    compute_trend,
    energy_growth,
    longest_falling_streak,
    longest_rising_streak,
    peak_density,
    remove_invalid,
    rolling_statistics,
    safe_divide,
    zero_crossing_rate,
)


class TemporalFeatureExtractor:
    """
    Extract temporal descriptors from
    astronomical time-series observations.
    """

    def extract(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, float | int]:

        ##################################################
        # CLEAN DATA
        ##################################################

        signal, timestamps = remove_invalid(
            signal,
            timestamps,
        )

        ##################################################
        # VALIDATION
        ##################################################

        if signal.size == 0:
            raise ValueError("Signal is empty.")

        if signal.size != timestamps.size:
            raise ValueError(
                "Signal and timestamps must have equal length."
            )

        features: dict[str, float | int] = {}

        ##################################################
        # TREND FEATURES
        ##################################################

        # Assumes compute_trend returns:
        # slope, intercept, r2
        slope, _, r2 = compute_trend(
            signal,
            timestamps,
        )

        features["trend_slope"] = float(slope)
        features["trend_r2"] = float(r2)

        ##################################################
        # AUTOCORRELATION
        ##################################################

        features["autocorr_lag1"] = float(
            autocorrelation(signal, lag=1)
        )

        features["autocorr_lag5"] = float(
            autocorrelation(signal, lag=5)
        )

        features["autocorr_lag10"] = float(
            autocorrelation(signal, lag=10)
        )

        ##################################################
        # ZERO CROSSING RATE
        ##################################################

        features["zero_crossing_rate"] = float(
            zero_crossing_rate(signal)
        )

        ##################################################
        # PEAK DENSITY
        ##################################################

        features["peak_density"] = float(
            peak_density(
                signal,
                timestamps,
            )
        )

        ##################################################
        # PERSISTENCE FEATURES
        ##################################################

        features["longest_rising_streak"] = int(
            longest_rising_streak(signal)
        )

        features["longest_falling_streak"] = int(
            longest_falling_streak(signal)
        )

        ##################################################
        # ENERGY GROWTH
        ##################################################

        features["energy_growth"] = float(
            energy_growth(signal)
        )

        ##################################################
        # ROLLING STATISTICS
        ##################################################

        # Assumes rolling_statistics returns:
        # rolling_mean, rolling_std, rolling_variance

        _, rolling_std, _ = rolling_statistics(
            signal
        )

        features["rolling_std"] = float(
            rolling_std
        )

        ##################################################
        # GRADIENT FEATURES
        ##################################################

        gradient = compute_gradient(
            signal,
            timestamps,
            smooth=True,
        )

        features["mean_absolute_gradient"] = float(
            np.mean(
                np.abs(gradient)
            )
        )

        ##################################################
        # SIGNAL DURATION
        ##################################################

        duration = max(
            float(timestamps[-1] - timestamps[0]),
            1e-10,
        )

        features["duration"] = duration

        ##################################################
        # ENERGY DENSITY
        ##################################################

        signal_energy = np.sum(signal ** 2)

        features["signal_energy_density"] = float(
            safe_divide(
                signal_energy,
                signal.size,
            )
        )

        ##################################################
        # COEFFICIENT OF VARIATION
        ##################################################

        features["coefficient_of_variation"] = float(
            safe_divide(
                np.std(signal),
                np.mean(signal),
            )
        )

        ##################################################
        # RETURN FEATURES
        ##################################################

        return features