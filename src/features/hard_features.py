"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
hard_features.py

Extract physics-informed features from
HEL1OS hard X-ray observations.
==========================================================
"""

from __future__ import annotations

import numpy as np

from scipy.signal import find_peaks
from scipy.stats import kurtosis
from scipy.stats import skew

from src.utils.feature_utils import (
    compute_gradient,
    remove_invalid,
    safe_divide,
)


class HardFeatureExtractor:
    """
    Extract statistical and burst-related
    features from HEL1OS hard X-ray data.
    """

    def __init__(
        self,
        peak_prominence_factor: float = 0.5,
        peak_distance: int = 5,
    ):
        self.peak_prominence_factor = peak_prominence_factor
        self.peak_distance = peak_distance

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

        features: dict[str, float] = {}

        ##################################################
        # BASIC STATISTICS
        ##################################################

        features["hard_mean"] = float(np.mean(signal))

        features["hard_median"] = float(np.median(signal))

        features["hard_max"] = float(np.max(signal))

        features["hard_std"] = float(np.std(signal))

        features["hard_rms"] = float(
            np.sqrt(np.mean(signal ** 2))
        )

        ##################################################
        # DISTRIBUTION
        ##################################################

        features["hard_skewness"] = float(
            skew(signal)
        )

        features["hard_kurtosis"] = float(
            kurtosis(signal)
        )

        ##################################################
        # ENERGY
        ##################################################

        features["hard_signal_energy"] = float(
            np.sum(signal ** 2)
        )

        integrated_flux = np.trapezoid(
            signal,
            timestamps,
        )

        features["hard_integrated_flux"] = float(
            integrated_flux
        )

        ##################################################
        # DYNAMICS
        ##################################################

        gradient = compute_gradient(
            signal,
            timestamps,
            smooth=True,
        )

        features["hard_rise_rate"] = float(
            np.max(gradient)
        )

        features["hard_decay_rate"] = float(
            abs(np.min(gradient))
        )

        ##################################################
        # PEAK FEATURES
        ##################################################

        prominence = (
            np.std(signal)
            * self.peak_prominence_factor
        )

        peaks, properties = find_peaks(
            signal,
            prominence=prominence,
            distance=self.peak_distance,
        )

        features["hard_num_peaks"] = int(len(peaks))

        if len(peaks):

            peak_values = signal[peaks]

            features["hard_average_peak_height"] = float(
                np.mean(peak_values)
            )

            features["hard_peak_prominence_mean"] = float(
                np.mean(
                    properties["prominences"]
                )
            )

        else:
            features["hard_average_peak_height"] = 0.0

            features["hard_peak_prominence_mean"] = 0.0

        ##################################################
        # BURST FACTOR
        ##################################################

        features["hard_burst_factor"] = safe_divide(
            features["hard_max"],
            features["hard_mean"],
        )

        return features