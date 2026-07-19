"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
feature_utils.py

Common utility functions used across
the feature engineering pipeline.
==========================================================
"""

from __future__ import annotations
import pywt

import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.fft import rfft
from scipy.fft import rfftfreq
from scipy.stats import entropy
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr


def remove_invalid(
    signal: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove NaN and Inf values.

    Parameters
    ----------
    signal : np.ndarray

    timestamps : np.ndarray

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
    """

    signal = np.asarray(signal, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    mask = (
        np.isfinite(signal)
        & np.isfinite(timestamps)
    )

    return signal[mask], timestamps[mask]

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe divide.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )

def smooth_signal(
    signal: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Smooth signal using Savitzky-Golay.
    """

    signal = np.asarray(signal, dtype=float)

    if signal.size < 5:
        return signal

    window = min(
        window_length,
        signal.size,
    )

    if window % 2 == 0:
        window -= 1

    if window <= polyorder:
        return signal

    return savgol_filter(
        signal,
        window_length=window,
        polyorder=polyorder,
    )

def compute_gradient(
    signal: np.ndarray,
    timestamps: np.ndarray,
    smooth: bool = True,
) -> np.ndarray:
    """
    Compute temporal gradient.
    """

    if smooth:
        signal = smooth_signal(signal)

    return np.gradient(
        signal,
        timestamps,
    )

def compute_trend(
    signal: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[float, float, float]:
    """
    Returns

    slope

    intercept

    r_squared
    """

    result = linregress(
        timestamps,
        signal,
    )

    return (
        float(result.slope),
        float(result.intercept),
        float(result.rvalue ** 2),
    )

def autocorrelation(
    signal: np.ndarray,
    lag: int,
) -> float:
    """
    Compute autocorrelation.
    """

    if lag <= 0:
        raise ValueError(
            "lag must be positive."
        )

    if lag >= signal.size:
        return 0.0

    x = signal[:-lag]

    y = signal[lag:]

    if (
        np.std(x) == 0
        or
        np.std(y) == 0
    ):
        return 0.0

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )

def zero_crossing_rate(
    signal: np.ndarray,
) -> float:
    """
    Compute zero crossing rate.
    """

    centered = signal - np.mean(signal)

    crossings = np.where(
        np.diff(
            np.sign(centered)
        )
    )[0]

    return float(
        len(crossings)
        /
        signal.size
    )

def peak_density(
    signal: np.ndarray,
    timestamps: np.ndarray,
    prominence_factor: float = 0.5,
    distance: int = 5,
) -> float:
    """
    Significant peaks
    per unit time.
    """

    prominence = (
        np.std(signal)
        *
        prominence_factor
    )

    peaks, _ = find_peaks(
        signal,
        prominence=prominence,
        distance=distance,
    )

    duration = (
        timestamps[-1]
        -
        timestamps[0]
    )

    return safe_divide(
        len(peaks),
        duration,
    )

def longest_rising_streak(
    signal: np.ndarray,
) -> int:
    """
    Longest increasing sequence.
    """

    longest = 1
    current = 1

    for i in range(
        1,
        signal.size,
    ):

        if signal[i] > signal[i - 1]:

            current += 1

            longest = max(
                longest,
                current,
            )

        else:

            current = 1

    return longest

def longest_falling_streak(
    signal: np.ndarray,
) -> int:
    """
    Longest decreasing sequence.
    """

    longest = 1
    current = 1

    for i in range(
        1,
        signal.size,
    ):

        if signal[i] < signal[i - 1]:

            current += 1

            longest = max(
                longest,
                current,
            )

        else:

            current = 1

    return longest

def energy_growth(
    signal: np.ndarray,
) -> float:
    """
    Energy(second half)
/ Energy(first half)
    """

    midpoint = signal.size // 2

    first_half = np.sum(
        signal[:midpoint] ** 2
    )

    second_half = np.sum(
        signal[midpoint:] ** 2
    )

    return safe_divide(
        second_half,
        first_half,
    )

def rolling_statistics(
    signal: np.ndarray,
    window: int = 20,
) -> tuple[float, float, float]:
    """
    Average rolling mean,
    std,
    variance.
    """

    signal = np.asarray(
        signal,
        dtype=float,
    )

    if signal.size < window:

        return (
            float(
                np.mean(signal)
            ),
            float(
                np.std(signal)
            ),
            float(
                np.var(signal)
            ),
        )

    rolling_means = []

    rolling_stds = []

    rolling_variances = []

    for i in range(
        signal.size - window + 1
    ):

        chunk = signal[
            i:i + window
        ]

        rolling_means.append(
            np.mean(chunk)
        )

        rolling_stds.append(
            np.std(chunk)
        )

        rolling_variances.append(
            np.var(chunk)
        )

    return (

        float(
            np.mean(
                rolling_means
            )
        ),

        float(
            np.mean(
                rolling_stds
            )
        ),

        float(
            np.mean(
                rolling_variances
            )
        ),
    )
    
def compute_fft(
    signal: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT.

    Returns
    -------
    frequencies
    magnitudes
    """

    signal = np.asarray(signal, dtype=float)
    
    signal = signal - np.mean(signal)
    
    signal = signal * np.hanning(signal.size)

    dt = np.mean(
        np.diff(timestamps)
    )

    fft = rfft(signal)

    frequencies = rfftfreq(
        signal.size,
        d=dt,
    )

    magnitudes = np.abs(fft)

    magnitudes = magnitudes / np.sum(magnitudes)

    return frequencies, magnitudes

def dominant_frequency(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
) -> float:
    """
    Dominant frequency.
    """

    if magnitudes.size == 0:
        return 0.0

    idx = np.argmax(magnitudes)

    return float(
        frequencies[idx]
    )
    
def spectral_centroid(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
) -> float:
    """
    Spectral centroid.
    """

    total = np.sum(magnitudes)

    if total == 0:
        return 0.0

    return float(
        np.sum(
            frequencies * magnitudes
        )
        /
        total
    )
    
def spectral_bandwidth(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
) -> float:
    """
    Spectral bandwidth.
    """

    centroid = spectral_centroid(
        frequencies,
        magnitudes,
    )

    total = np.sum(magnitudes)

    if total == 0:
        return 0.0

    return float(

        np.sqrt(

            np.sum(

                (
                    frequencies
                    -
                    centroid
                ) ** 2
                *
                magnitudes

            )

            /

            total

        )

    )
    
def spectral_entropy(
    magnitudes: np.ndarray,
) -> float:
    """
    Spectral entropy.
    """

    power = magnitudes ** 2

    total = np.sum(power)

    if total == 0:
        return 0.0

    power /= total

    return float(
        entropy(
            power
        )
    )
    
def spectral_energy(
    magnitudes: np.ndarray,
) -> float:
    """
    Spectral energy.
    """

    return float(
        np.sum(
            magnitudes ** 2
        )
    )
    
def low_high_frequency_ratio(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    Low vs High frequency energy.
    """

    power = magnitudes ** 2

    low = np.sum(
        power[
            frequencies <= threshold
        ]
    )

    high = np.sum(
        power[
            frequencies > threshold
        ]
    )

    return safe_divide(
        low,
        high,
    )
    
def compute_wavelet(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> list[np.ndarray]:
    """
    Compute Discrete Wavelet Transform (DWT).

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    wavelet : str
        Wavelet family.

    level : int
        Decomposition level.

    Returns
    -------
    list[np.ndarray]
        Wavelet coefficients.
    """

    signal = np.asarray(signal, dtype=float)

    return pywt.wavedec(
        signal,
        wavelet=wavelet,
        level=level,
    )

def wavelet_energy(
    coeffs: list[np.ndarray],
) -> list[float]:
    """
    Energy of each wavelet coefficient.
    """

    return [
        float(np.sum(c ** 2))
        for c in coeffs
    ]

def wavelet_entropy(
    energies: list[float],
) -> float:
    """
    Shannon entropy of wavelet energies.
    """

    energies = np.asarray(
        energies,
        dtype=float,
    )

    total = np.sum(energies)

    if total == 0:
        return 0.0

    probabilities = energies / total

    return float(

        -np.sum(

            probabilities
            *
            np.log2(
                probabilities + 1e-12
            )

        )

    )
    
def total_wavelet_energy(
    energies: list[float],
) -> float:
    """
    Total wavelet energy.
    """

    return float(
        np.sum(energies)
    )
    
def dominant_wavelet_scale(
    energies: list[float],
) -> int:
    """
    Wavelet level with maximum energy.
    """

    return int(
        np.argmax(energies)
    )

def shannon_entropy(
    signal: np.ndarray,
    bins: int = 64,
) -> float:
    """
    Compute Shannon entropy of a signal.
    """

    hist, _ = np.histogram(
        signal,
        bins=bins,
        density=True,
    )

    hist = hist[hist > 0]

    if hist.size == 0:
        return 0.0

    return float(
        entropy(hist)
    )
    
def energy_entropy(
    signal: np.ndarray,
) -> float:
    """
    Shannon entropy of signal energy.
    """

    energy = signal ** 2

    total = np.sum(energy)

    if total == 0:
        return 0.0

    probability = energy / total

    return float(
        -np.sum(
            probability
            *
            np.log2(
                probability + 1e-12
            )
        )
    )
    
def histogram_entropy(
    signal: np.ndarray,
    bins: int = 32,
) -> float:
    """
    Histogram-based entropy.
    """

    hist, _ = np.histogram(
        signal,
        bins=bins,
    )

    hist = hist.astype(float)

    hist /= np.sum(hist)

    hist = hist[hist > 0]

    return float(
        entropy(hist)
    )

def approximate_entropy(
    signal: np.ndarray,
) -> float:
    """
    Approximate entropy (simplified).

    Uses signal variance as tolerance.
    """

    signal = np.asarray(signal)

    if signal.size < 20:
        return 0.0

    tolerance = 0.2 * np.std(signal)

    matches = 0

    total = 0

    for i in range(signal.size - 1):

        for j in range(i + 1, signal.size - 1):

            if abs(signal[i] - signal[j]) < tolerance:

                matches += 1

            total += 1

    return safe_divide(
        matches,
        total,
    )
    
def pearson_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
) -> float:
    """
    Compute Pearson correlation coefficient.
    """

    if signal1.size != signal2.size:
        raise ValueError(
            "Signals must have equal length."
        )

    return float(
        pearsonr(signal1, signal2)[0]
    )
    
def spearman_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
) -> float:
    """
    Compute Spearman rank correlation.
    """

    if signal1.size != signal2.size:
        raise ValueError(
            "Signals must have equal length."
        )

    return float(
        spearmanr(signal1, signal2)[0]
    )
    
def cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized cross-correlation.
    """

    s1 = signal1 - np.mean(signal1)
    s2 = signal2 - np.mean(signal2)

    return correlate(
        s1,
        s2,
        mode="full",
    )
    
def max_cross_correlation(
    correlation: np.ndarray,
) -> float:
    """
    Maximum cross-correlation value.
    """

    return float(
        np.max(correlation)
    )
    
def lag_at_max_correlation(
    correlation: np.ndarray,
    signal_length: int,
) -> int:
    """
    Lag corresponding to maximum cross-correlation.
    """

    index = np.argmax(correlation)

    return int(
        index - (signal_length - 1)
    )
    
def rms_difference(
    signal1: np.ndarray,
    signal2: np.ndarray,
) -> float:
    """
    Root Mean Square difference.
    """

    return float(
        np.sqrt(
            np.mean(
                (signal1 - signal2) ** 2
            )
        )
    )
    
