"""
src/features/rolling_feature_extractor.py
──────────────────────────────────────────
Temporal rolling-window feature extraction for production TCN input.

The Problem Being Solved
------------------------
The original feature extractors (soft_features.py, hard_features.py, etc.)
compute ONE scalar per feature per ENTIRE observation (e.g. np.mean(signal)).
This produces a DataFrame with shape:

    (N_observations, F)   e.g. (49, 38)      ← WRONG for TCN

The production TCN requires the feature matrix to have shape:

    (T_total_timesteps, F)  e.g. (400000+, 38)  ← CORRECT for TCN

Where each row is one second of solar observation and each column is a
physics-informed feature computed from a short rolling context window
centred at that second.

Approach
--------
For each second t in the observation, we take a short context window
[t - half_window, t + half_window] of the raw signals and compute the
same features the original extractors computed — but for that local
window only. This produces one row per second.

Features computed per second (same set as original pipeline → F=38):
  Soft X-ray (SoLEXS COUNTS):
    soft_mean, soft_median, soft_max, soft_std, soft_rms,
    soft_skewness, soft_kurtosis, soft_signal_energy, soft_integrated_flux,
    soft_rise_rate, soft_decay_rate, soft_num_peaks, soft_average_peak_height

  Hard X-ray (HEL1OS energy):
    hard_mean, hard_median, hard_max, hard_std, hard_rms,
    hard_skewness, hard_kurtosis, hard_signal_energy, hard_integrated_flux,
    hard_rise_rate, hard_decay_rate, hard_num_peaks,
    hard_average_peak_height, hard_peak_prominence_mean, hard_burst_factor

  Temporal:
    trend_slope, trend_r2, autocorr_lag1, autocorr_lag5, autocorr_lag10,
    zero_crossing_rate, peak_density, energy_growth,
    rolling_std, mean_absolute_gradient, duration, signal_energy_density,
    coefficient_of_variation, longest_rising_streak, longest_falling_streak

  Cross-channel:
    cross_correlation, soft_hard_ratio
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger("helioforge.features.rolling")

# NumPy 2.2 removed np.trapz — use np.trapezoid with a fallback for older numpy
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
if _trapz is None:
    def _trapz(y, x=None):  # type: ignore[misc]
        """Pure-numpy trapezoidal integration fallback."""
        if x is None:
            return float(np.sum(y[:-1] + y[1:]) * 0.5)
        dx = np.diff(x)
        return float(np.sum((y[:-1] + y[1:]) * 0.5 * dx))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fast numpy skew / kurtosis (no scipy overhead per call)
# ─────────────────────────────────────────────────────────────────────────────

def _skew_np(x: np.ndarray) -> float:
    """Fisher skewness — matches scipy.stats.skew(bias=True)."""
    if len(x) < 3:
        return 0.0
    mu = np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 3))


def _kurt_np(x: np.ndarray) -> float:
    """Excess kurtosis — matches scipy.stats.kurtosis(fisher=True)."""
    if len(x) < 4:
        return 0.0
    mu = np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 4)) - 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe scalar functions
# ─────────────────────────────────────────────────────────────────────────────

def _safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    return float(a / b) if abs(b) > 1e-10 else fallback


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return 0.0
    mu = np.mean(x)
    denom = np.sum((x - mu) ** 2)
    if denom < 1e-12:
        return 0.0
    num = np.sum((x[:-lag] - mu) * (x[lag:] - mu))
    return float(num / denom)


def _zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    centered = x - np.mean(x)
    crossings = np.sum(np.diff(np.sign(centered)) != 0)
    return float(crossings / max(len(x) - 1, 1))


def _longest_streak(x: np.ndarray, rising: bool) -> int:
    if len(x) < 2:
        return 0
    diffs = np.diff(x)
    mask = diffs > 0 if rising else diffs < 0
    best = current = 0
    for v in mask:
        if v:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _energy_growth(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    half = len(x) // 2
    e1 = float(np.sum(x[:half] ** 2))
    e2 = float(np.sum(x[half:] ** 2))
    return _safe_divide(e2 - e1, e1 + 1e-12)


def _trend(x: np.ndarray, t: np.ndarray):
    """Linear regression — returns (slope, intercept, r2)."""
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    t_c = t - np.mean(t)
    x_c = x - np.mean(x)
    denom = np.sum(t_c ** 2)
    if denom < 1e-12:
        return 0.0, float(np.mean(x)), 0.0
    slope = float(np.sum(t_c * x_c) / denom)
    intercept = float(np.mean(x) - slope * np.mean(t))
    residuals = x - (slope * t + intercept)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum(x_c ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return slope, intercept, r2


def _extract_window_features(
    soft_w: np.ndarray,
    hard_w: np.ndarray,
    ts_w: np.ndarray,
    peak_prominence_factor: float = 0.5,
    peak_distance: int = 5,
) -> Dict[str, float]:
    """
    Compute all F features from one short window of soft + hard signals.
    Returns a flat dict of scalars.
    """
    feat: Dict[str, float] = {}

    # ── Soft features ────────────────────────────────────────────────────────
    s = soft_w
    feat["soft_mean"]            = float(np.mean(s))
    feat["soft_median"]          = float(np.median(s))
    feat["soft_max"]             = float(np.max(s))
    feat["soft_std"]             = float(np.std(s))
    feat["soft_rms"]             = float(np.sqrt(np.mean(s ** 2)))
    feat["soft_skewness"]        = _skew_np(s)
    feat["soft_kurtosis"]        = _kurt_np(s)
    feat["soft_signal_energy"]   = float(np.sum(s ** 2))
    feat["soft_integrated_flux"] = _trapz(s, ts_w)

    s_grad = np.gradient(s, ts_w) if len(s) > 1 else np.zeros(1)
    feat["soft_rise_rate"]  = float(np.max(s_grad))
    feat["soft_decay_rate"] = float(abs(np.min(s_grad)))

    s_prom = np.std(s) * peak_prominence_factor
    s_peaks, _ = find_peaks(s, prominence=max(s_prom, 1e-10), distance=peak_distance)
    feat["soft_num_peaks"]           = int(len(s_peaks))
    feat["soft_average_peak_height"] = (
        float(np.mean(s[s_peaks])) if len(s_peaks) > 0 else 0.0
    )

    # ── Hard features ────────────────────────────────────────────────────────
    h = hard_w
    feat["hard_mean"]            = float(np.mean(h))
    feat["hard_median"]          = float(np.median(h))
    feat["hard_max"]             = float(np.max(h))
    feat["hard_std"]             = float(np.std(h))
    feat["hard_rms"]             = float(np.sqrt(np.mean(h ** 2)))
    feat["hard_skewness"]        = _skew_np(h)
    feat["hard_kurtosis"]        = _kurt_np(h)
    feat["hard_signal_energy"]   = float(np.sum(h ** 2))
    feat["hard_integrated_flux"] = _trapz(h, ts_w)

    h_grad = np.gradient(h, ts_w) if len(h) > 1 else np.zeros(1)
    feat["hard_rise_rate"]  = float(np.max(h_grad))
    feat["hard_decay_rate"] = float(abs(np.min(h_grad)))

    h_prom = np.std(h) * peak_prominence_factor
    h_peaks, h_props = find_peaks(h, prominence=max(h_prom, 1e-10), distance=peak_distance)
    feat["hard_num_peaks"]             = int(len(h_peaks))
    feat["hard_average_peak_height"]   = (
        float(np.mean(h[h_peaks])) if len(h_peaks) > 0 else 0.0
    )
    feat["hard_peak_prominence_mean"]  = (
        float(np.mean(h_props["prominences"])) if len(h_peaks) > 0 else 0.0
    )
    feat["hard_burst_factor"] = _safe_divide(feat["hard_max"], feat["hard_mean"])

    # ── Temporal features (computed on soft) ─────────────────────────────────
    slope, _, r2 = _trend(s, ts_w)
    feat["trend_slope"]                = slope
    feat["trend_r2"]                   = r2
    feat["autocorr_lag1"]              = _autocorr(s, 1)
    feat["autocorr_lag5"]              = _autocorr(s, 5)
    feat["autocorr_lag10"]             = _autocorr(s, 10)
    feat["zero_crossing_rate"]         = _zero_crossing_rate(s)
    t_range = max(float(ts_w[-1] - ts_w[0]), 1e-10)
    feat["peak_density"]               = _safe_divide(len(s_peaks), t_range)
    feat["energy_growth"]              = _energy_growth(s)

    # rolling_std: std of the second half vs first half (proxy for rolling)
    half = max(len(s) // 2, 1)
    feat["rolling_std"]                = float(np.std(s[-half:]))
    feat["mean_absolute_gradient"]     = float(np.mean(np.abs(s_grad)))
    feat["duration"]                   = t_range
    feat["signal_energy_density"]      = _safe_divide(float(np.sum(s ** 2)), len(s))
    feat["coefficient_of_variation"]   = _safe_divide(float(np.std(s)), float(np.mean(s)))
    feat["longest_rising_streak"]      = float(_longest_streak(s, rising=True))
    feat["longest_falling_streak"]     = float(_longest_streak(s, rising=False))

    # ── Cross-channel features ────────────────────────────────────────────────
    if np.std(s) > 1e-10 and np.std(h) > 1e-10:
        feat["cross_correlation"] = float(np.corrcoef(s, h)[0, 1])
    else:
        feat["cross_correlation"] = 0.0

    feat["soft_hard_ratio"] = _safe_divide(float(np.mean(s)), float(np.mean(h)))

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class RollingFeatureExtractor:
    """
    Converts a raw 1 Hz observation (soft_signal, hard_signal, timestamps)
    into a per-second feature matrix of shape (T, F).

    Parameters
    ----------
    context_seconds : int
        Length of the rolling context window in seconds (default 60).
        At each second t, features are computed from the signal segment
        [t - context_seconds//2, t + context_seconds//2].
        Must be >= 10 to give meaningful statistics.
    stride : int
        Step between successive output rows in seconds (default 1).
        stride=1 → one row per second (maximum resolution).
        stride=32 → one row every 32 seconds (faster, sparser).
    """

    def __init__(
        self,
        context_seconds: int = 60,
        stride: int = 32,
    ) -> None:
        if context_seconds < 10:
            raise ValueError("context_seconds must be >= 10 for meaningful statistics.")
        self.context_seconds = context_seconds
        self.stride = stride

    def extract(
        self,
        soft_signal: np.ndarray,
        hard_signal: np.ndarray,
        timestamps: np.ndarray,
        observation_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        soft_signal : np.ndarray  shape (T,)
        hard_signal : np.ndarray  shape (T,)
        timestamps  : np.ndarray  shape (T,)
        observation_id : str, optional
            Added as a column for traceability.

        Returns
        -------
        pd.DataFrame
            Shape (T', F+1) where T' = number of strided timesteps,
            F = number of features, and the extra column is 'TIME'.
        """
        T = len(soft_signal)
        if T < self.context_seconds:
            logger.warning(
                "Observation too short: T=%d samples < context=%d seconds. "
                "Either this observation is genuinely short, or run with "
                "--context %d to process it.",
                T, self.context_seconds, max(T // 2, 2)
            )
            print(
                f"         → SKIP (T={T} samples < context={self.context_seconds}s — "
                f"try --context {max(T // 2, 2)})"
            )
            return pd.DataFrame()

        half = self.context_seconds // 2
        rows: List[Dict] = []

        # Centre the window at each strided index
        for t in range(half, T - half, self.stride):
            start = t - half
            end   = t + half

            s_w  = soft_signal[start:end].astype(np.float64)
            h_w  = hard_signal[start:end].astype(np.float64)
            ts_w = timestamps[start:end].astype(np.float64)

            # Sanitize
            mask = np.isfinite(s_w) & np.isfinite(h_w) & np.isfinite(ts_w)
            if mask.sum() < 10:
                continue

            s_w, h_w, ts_w = s_w[mask], h_w[mask], ts_w[mask]

            try:
                feat = _extract_window_features(s_w, h_w, ts_w)
            except Exception as exc:
                logger.warning("Window t=%d failed: %s — skipping window", t, exc)
                continue

            feat["TIME"] = float(timestamps[t])
            if observation_id is not None:
                feat["observation_id"] = observation_id
            rows.append(feat)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Put TIME first
        cols = ["TIME"] + [c for c in df.columns if c not in ("TIME", "observation_id")]
        if "observation_id" in df.columns:
            cols.append("observation_id")
        return df[cols]
