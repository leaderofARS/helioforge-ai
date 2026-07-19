"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
soft_features.py

Extract physics-informed statistical features from
SoLEXS soft X-ray light curves.
==========================================================
"""

from __future__ import annotations

import numpy as np

from scipy.signal import find_peaks
from scipy.stats import kurtosis
from scipy.stats import skew

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


class SoftFeatureExtractor:
    """
    Extract statistical and physical features
    from a SoLEXS light curve.
    """

    def __init__(
    self,
    peak_prominence_factor: float = 0.5,
    peak_distance: int = 5,
    ) -> None:
        self.peak_prominence_factor = peak_prominence_factor
        self.peak_distance = peak_distance

    def extract(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray
    ) -> dict[str, float | int]:
        """
        Parameters
        ----------
        signal : np.ndarray
            Soft X-ray intensity values.

        timestamps : np.ndarray
            Observation timestamps.

        Returns
        -------
        dict
            Dictionary containing extracted features.
        """
        mask = np.isfinite(signal) & np.isfinite(timestamps)
        signal = signal[mask]
        timestamps = timestamps[mask]

        if len(signal) == 0:
            raise ValueError("Signal is empty.")

        if len(signal) != len(timestamps):
            raise ValueError(
                "Signal and timestamps must have equal length."
            )

        features = {}

        ##################################################
        # BASIC STATISTICS
        ##################################################

        features["soft_mean"] = float(np.mean(signal))

        features["soft_median"] = float(np.median(signal))

        features["soft_max"] = float(np.max(signal))

        features["soft_std"] = float(np.std(signal))

        features["soft_rms"] = float(np.sqrt(
            np.mean(signal ** 2)
        ))

        ##################################################
        # DISTRIBUTION
        ##################################################

        features["soft_skewness"] = float(skew(signal))

        features["soft_kurtosis"] = float(kurtosis(signal))

        ##################################################
        # ENERGY
        ##################################################

        features["soft_signal_energy"] = float(np.sum(signal ** 2))

        integrated_flux = float(np.trapz(signal, timestamps))

        features["soft_integrated_flux"] = integrated_flux

        ##################################################
        # DYNAMICS
        ##################################################

        gradient = np.gradient(signal, timestamps)

        features["soft_rise_rate"] = float(np.max(gradient))

        features["soft_decay_rate"] = float(abs(
            np.min(gradient)
        ))

        ##################################################
        # PEAKS
        ##################################################

        prominence = np.std(signal) * self.peak_prominence_factor
        peaks, _ = find_peaks(signal, prominence=prominence, distance=self.peak_distance)

        features["soft_num_peaks"] = int(len(peaks))

        if peaks.size > 0:

            peak_values = signal[peaks]

            features["soft_average_peak_height"] = float(np.mean(
                peak_values
            ))

        else:

            features["soft_average_peak_height"] = 0.0

        return features