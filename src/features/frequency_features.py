"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

frequency_features.py

Extract frequency-domain features using FFT.

Author: HelioForge AI
==========================================================
"""

from __future__ import annotations

import numpy as np

from src.utils.feature_utils import (
    compute_fft,
    dominant_frequency,
    low_high_frequency_ratio,
    remove_invalid,
    spectral_bandwidth,
    spectral_centroid,
    spectral_energy,
    spectral_entropy,
)


class FrequencyFeatureExtractor:
    """
    Extract FFT-based spectral features
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
        # VALIDATION
        ##################################################

        if signal.size == 0:
            raise ValueError(
                "Signal is empty."
            )

        if signal.size != timestamps.size:
            raise ValueError(
                "Signal and timestamps "
                "must have equal length."
            )

        ##################################################
        # FFT
        ##################################################

        frequencies, magnitudes = compute_fft(
            signal,
            timestamps,
        )

        features: dict[str, float] = {}

        ##################################################
        # DOMINANT FREQUENCY
        ##################################################

        features["dominant_frequency"] = float(
            dominant_frequency(
                frequencies,
                magnitudes,
            )
        )

        ##################################################
        # SPECTRAL CENTROID
        ##################################################

        features["spectral_centroid"] = float(
            spectral_centroid(
                frequencies,
                magnitudes,
            )
        )

        ##################################################
        # SPECTRAL BANDWIDTH
        ##################################################

        features["spectral_bandwidth"] = float(
            spectral_bandwidth(
                frequencies,
                magnitudes,
            )
        )

        ##################################################
        # SPECTRAL ENTROPY
        ##################################################

        features["spectral_entropy"] = float(
            spectral_entropy(
                magnitudes,
            )
        )
        
        ##################################################
        # SPECTRAL ENERGY
        ##################################################

        features["spectral_energy"] = float(
            spectral_energy(
                magnitudes,
            )
        )

        ##################################################
        # LOW / HIGH FREQUENCY RATIO
        ##################################################

        features["low_high_frequency_ratio"] = float(
            low_high_frequency_ratio(
                frequencies,
                magnitudes,
            )
        )

        ##################################################
        # PEAK MAGNITUDE
        ##################################################

        features["peak_frequency_magnitude"] = float(
            np.max(magnitudes)
        )

        ##################################################
        # SPECTRUM STANDARD DEVIATION
        ##################################################

        features["spectrum_std"] = float(
            np.std(magnitudes)
        )
        
        ##################################################
        # SPECTRUM DYNAMIC RANGE
        ##################################################

        features["spectrum_dynamic_range"] = float(
            np.max(magnitudes)
            - np.min(magnitudes)
        )

        ##################################################
        # SPECTRAL ROLLOFF (85%)
        ##################################################

        power = magnitudes ** 2

        cumulative_power = np.cumsum(power)

        total_power = cumulative_power[-1]
        
        if total_power == 0:
            features["spectral_rolloff"] = 0.0
        else:
            rolloff_index = np.searchsorted(
                cumulative_power,
                0.85 * total_power,
            )

            features["spectral_rolloff"] = float(
                frequencies[
                    min(
                        rolloff_index,
                        len(frequencies) - 1,
                    )
                ]
            )

        ##################################################
        # SPECTRAL FLATNESS
        ##################################################

        eps = 1e-12

        geometric_mean = np.exp(
            np.mean(
                np.log(
                    magnitudes + eps
                )
            )
        )

        arithmetic_mean = np.mean(
            magnitudes + eps
        )

        features["spectral_flatness"] = float(
            geometric_mean
            / arithmetic_mean
        )

        ##################################################
        # RETURN FEATURES
        ##################################################

        return features