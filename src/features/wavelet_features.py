"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
wavelet_features.py

Extract wavelet-based features from
astronomical time-series observations.
==========================================================
"""

from __future__ import annotations

import numpy as np

from src.utils.feature_utils import (
    compute_wavelet,
    dominant_wavelet_scale,
    remove_invalid,
    total_wavelet_energy,
    wavelet_energy,
    wavelet_entropy,
)


class WaveletFeatureExtractor:
    """
    Extract wavelet-domain features
    from astronomical time-series.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
    ):
        self.wavelet = wavelet
        self.level = level

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
        # COMPUTE WAVELET
        ##################################################

        coeffs = compute_wavelet(
            signal,
            wavelet=self.wavelet,
            level=self.level,
        )

        energies = wavelet_energy(coeffs)
        
        if len(energies) == 0:
            raise ValueError("Wavelet energy computation failed.")

        features: dict[str, float] = {}

        ##################################################
        # TOTAL ENERGY
        ##################################################

        features["wavelet_total_energy"] = float(
            total_wavelet_energy(
                energies
            )
        )

        ##################################################
        # WAVELET ENTROPY
        ##################################################

        features["wavelet_entropy"] = float(
            wavelet_entropy(
                energies
            )
        )

        ##################################################
        # DOMINANT SCALE
        ##################################################

        features["dominant_wavelet_scale"] = float(
            dominant_wavelet_scale(
                energies
            )
        )
        
        ##################################################
        # ENERGY AT EACH LEVEL
        ##################################################

        for i, energy in enumerate(energies):

            features[f"wavelet_energy_level_{i}"] = (
                float(energy)
            )

        ##################################################
        # COEFFICIENT STATISTICS
        ##################################################

        approximation = coeffs[0]

        features["wavelet_std"] = float(
            np.std(approximation)
        )

        features["wavelet_rms"] = float(
            np.sqrt(
                np.mean(
                    approximation ** 2
                )
            )
        )

        ##################################################
        # DETAIL COEFFICIENT STATISTICS
        ##################################################

        detail_coeffs = np.concatenate(
            coeffs[1:]
        )


        features["detail_std"] = float(
            np.std(detail_coeffs)
        )

        features["detail_energy"] = float(
            np.sum(
                detail_coeffs ** 2
            )
        )

        ##################################################
        # RETURN FEATURES
        ##################################################

        return features