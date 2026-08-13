"""Upload preprocessing for dashboard observations.

The trained TCN consumes a normalized (1, 32, 512) tensor.  FITS products
vary by instrument, so this module deliberately accepts the first numeric FITS
image/table column and creates the same fixed input contract for inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

WINDOW = 512
FEATURE_NAMES = [
    "soft_intensity", "hard_intensity", "signal_gradient", "signal_curvature",
    "rolling_mean_8", "rolling_std_8", "rolling_mean_32", "rolling_std_32",
    "rolling_mean_64", "rolling_std_64", "rolling_min_32", "rolling_max_32",
    "rolling_range_32", "rolling_zscore", "rolling_energy_32", "peak_prominence",
    "low_frequency_energy", "mid_frequency_energy", "high_frequency_energy", "spectral_centroid",
    "wavelet_energy_l1", "wavelet_energy_l2", "wavelet_energy_l3", "entropy_proxy",
    "positive_gradient", "negative_gradient", "rolling_median_32", "rolling_iqr_32",
    "signal_variance_64", "signal_skew_proxy", "signal_kurtosis_proxy", "activity_index",
]


@dataclass
class PreparedObservation:
    tensor: torch.Tensor
    signal: list[float]
    features: dict[str, float]
    rgb_intensity: dict[str, int]
    active_regions: list[dict]


def _resize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("Observation has fewer than two finite numeric samples.")
    source = np.linspace(0, 1, values.size)
    return np.interp(np.linspace(0, 1, WINDOW), source, values)


def _normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros(WINDOW, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0, 1).astype(np.float32)


def _rolling(values: np.ndarray, size: int, operation: str) -> np.ndarray:
    padded = np.pad(values, (size - 1, 0), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    if operation == "mean": return windows.mean(axis=-1)
    if operation == "std": return windows.std(axis=-1)
    if operation == "min": return windows.min(axis=-1)
    if operation == "max": return windows.max(axis=-1)
    if operation == "median": return np.median(windows, axis=-1)
    if operation == "energy": return np.mean(windows ** 2, axis=-1)
    raise ValueError(f"Unknown rolling operation: {operation}")


def _feature_tensor(signal: np.ndarray) -> np.ndarray:
    gradient = np.gradient(signal)
    curvature = np.gradient(gradient)
    mean8, std8 = _rolling(signal, 8, "mean"), _rolling(signal, 8, "std")
    mean32, std32 = _rolling(signal, 32, "mean"), _rolling(signal, 32, "std")
    mean64, std64 = _rolling(signal, 64, "mean"), _rolling(signal, 64, "std")
    minimum, maximum = _rolling(signal, 32, "min"), _rolling(signal, 32, "max")
    median = _rolling(signal, 32, "median")
    q75 = _rolling(signal, 32, "max") - _rolling(signal, 32, "min")
    centered = signal - mean32
    zscore = centered / (std32 + 1e-6)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    bands = np.array_split(spectrum, 3)
    band_energy = [np.full(WINDOW, float(np.mean(b ** 2))) for b in bands]
    centroid = np.full(WINDOW, float(np.dot(np.arange(len(spectrum)), spectrum) / (spectrum.sum() + 1e-6) / len(spectrum)))
    # Multiscale differences are robust equivalents when wavelet coefficients are unavailable.
    wavelets = [np.abs(signal - _rolling(signal, n, "mean")) for n in (4, 16, 64)]
    entropy = -signal * np.log(signal + 1e-6)
    variance64 = _rolling((signal - mean64) ** 2, 64, "mean")
    skew = centered ** 3 / (std32 ** 3 + 1e-6)
    kurtosis = centered ** 4 / (std32 ** 4 + 1e-6)
    channels = [signal, mean8, gradient, curvature, mean8, std8, mean32, std32, mean64, std64, minimum, maximum, maximum-minimum, zscore, _rolling(signal,32,"energy"), np.abs(gradient), *band_energy, centroid, *wavelets, entropy, np.maximum(gradient,0), np.maximum(-gradient,0), median, q75, variance64, skew, kurtosis, mean32 + std32]
    return np.stack([_normalize(channel) for channel in channels[:32]]).astype(np.float32)


def _regions(signal: np.ndarray) -> list[dict]:
    indices = np.argpartition(signal, -min(3, len(signal)))[-3:]
    return [{"id": f"AR-{i + 1:03d}", "lat": round(-25 + int(index) / WINDOW * 50, 1), "lon": round(-75 + int(index) / WINDOW * 150, 1), "class": 2 if value < .7 else 3, "confidence": round(float(value), 3), "intensity": round(float(value * 1000), 1)} for i, (index, value) in enumerate(sorted(zip(indices, signal[indices]), key=lambda item: item[1], reverse=True))]


def prepare_array(values: np.ndarray) -> PreparedObservation:
    raw = _resize(values)
    signal = _normalize(raw)
    channels = _feature_tensor(signal)
    features = {name: round(float(channels[i].mean()), 5) for i, name in enumerate(FEATURE_NAMES)}
    rgb = {"red": int(np.clip(raw.max(), 0, 255)), "green": int(np.clip(raw.mean() * 1.1, 0, 255)), "blue": int(np.clip(raw.std() * 2, 0, 255))}
    return PreparedObservation(torch.from_numpy(channels).unsqueeze(0), signal.tolist(), features, rgb, _regions(signal))


def prepare_upload(filename: str, content: bytes) -> PreparedObservation:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pt":
        value = torch.load(BytesIO(content), map_location="cpu", weights_only=False)
        tensor = value.get("sequences", value) if isinstance(value, dict) else value
        if not isinstance(tensor, torch.Tensor): raise ValueError("The .pt upload does not contain a tensor.")
        if tensor.ndim == 3: tensor = tensor[0]
        if tuple(tensor.shape) != (32, WINDOW): raise ValueError("Expected a .pt tensor with shape (32, 512) or (N, 32, 512).")
        signal = tensor[0].float().numpy()
        prepared = prepare_array(signal)
        prepared.tensor = tensor.float().unsqueeze(0)
        return prepared
    if suffix not in {".fits", ".fit", ".fts"}: raise ValueError("Upload a FITS (.fits/.fit/.fts) or normalized .pt observation.")
    try:
        from astropy.io import fits
        with fits.open(BytesIO(content), memmap=False) as hdul:
            arrays = [np.asarray(hdu.data) for hdu in hdul if hdu.data is not None and np.asarray(hdu.data).dtype.names is None]
            if not arrays: raise ValueError("FITS file has no numeric image extension.")
            return prepare_array(arrays[0])
    except ImportError as exc: raise RuntimeError("FITS support requires astropy.") from exc
