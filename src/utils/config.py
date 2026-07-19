"""
src/utils/config.py
────────────────────
Single source of truth for all project paths.

Loads configs/data_paths.yaml via PathConfig and exposes:
  - PATH_CFG  — typed PathConfig dataclass (primary interface)
  - get_path() — backward-compatible helper for string-keyed lookups

All paths are absolute (EC2 Ubuntu: /opt/helioforge/...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.HPINA.configs.paths import PathConfig

# ─── Locate the YAML relative to this file ──────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parents[2]
_PATHS_YAML  = _REPO_ROOT / "configs" / "data_paths.yaml"

# ─── Typed config (primary interface) ───────────────────────────────────────
PATH_CFG: PathConfig = PathConfig.from_yaml(_PATHS_YAML)

# ─── Flat alias map for backward-compatible get_path() ──────────────────────
_PATH_MAP: dict[str, Path] = {
    # raw
    "raw":            PATH_CFG.raw.root,
    "solexs":         PATH_CFG.raw.solexs,
    "hel1os":         PATH_CFG.raw.hel1os,
    # preprocessing stages
    "validated":      PATH_CFG.preprocessing.validated,
    "synchronized":   PATH_CFG.preprocessing.synchronized,
    "processed":      PATH_CFG.preprocessing.processed,
    # features
    "features":       PATH_CFG.features.root,
    # windows
    "windows":        PATH_CFG.windows.root,
    # metadata
    "metadata":       PATH_CFG.metadata.root,
    # models
    "models":         PATH_CFG.models.root,
    "baseline_tcn":   PATH_CFG.models.baseline_tcn,
    # experiments
    "experiments":    PATH_CFG.experiments.root,
    "runs":           PATH_CFG.experiments.baseline_tcn.runs,
    "checkpoints":    PATH_CFG.experiments.baseline_tcn.checkpoints,
    "ablations":      PATH_CFG.experiments.baseline_tcn.ablations,
    # outputs
    "outputs":        PATH_CFG.outputs.root,
    "predictions":    PATH_CFG.outputs.predictions,
    # reports & figures
    "reports":        PATH_CFG.reports.root,
    "figures":        PATH_CFG.reports.figures,
    "tables":         PATH_CFG.reports.tables,
    "visualizations": PATH_CFG.reports.figures,
    "distributions":  PATH_CFG.reports.figures / "distributions",
    # logs
    "logs":           PATH_CFG.logs.root,
}

# ─── File alias map ──────────────────────────────────────────────────────────
_FILE_MAP: dict[str, Path] = {
    "solexs_metadata":           PATH_CFG.metadata.solexs_metadata,
    "hel1os_metadata":           PATH_CFG.metadata.hel1os_metadata,
    "sync_report":               PATH_CFG.metadata.sync_report,
    "synchronization_report":    PATH_CFG.metadata.sync_report,
    "gti_table":                 PATH_CFG.metadata.gti_table,
    "features_csv":              PATH_CFG.features.csv,
    "features_parquet":          PATH_CFG.features.parquet,
    "features_excel":            PATH_CFG.features.excel,
    "normalisation_stats":       PATH_CFG.normalisation.stats_json,
    # observation-level filenames (relative, not absolute)
    "observation_lightcurve":    Path("lightcurve.csv"),
    "observation_event":         Path("event.csv"),
    # visualization output file aliases
    "feature_count_png":             Path("feature_count.png"),
    "feature_selection_summary_csv": Path("feature_selection_summary.csv"),
    "feature_selection_summary_png": Path("feature_selection_summary.png"),
    "missing_values_csv":            Path("missing_values_summary.csv"),
    "missing_values_png":            Path("missing_values.png"),
    "pca_summary_csv":               Path("pca_summary.csv"),
    "pca_explained_variance_png":    Path("pca_explained_variance.png"),
    "pca_cumulative_variance_png":   Path("pca_cumulative_variance.png"),
    "pca_projection_png":            Path("pca_projection.png"),
    "correlation_heatmap_png":       Path("correlation_heatmap.png"),
}

# ─── Export aliases ──────────────────────────────────────────────────────────
_EXPORT_MAP: dict[str, Any] = {
    "csv":     {"directory": str(PATH_CFG.features.root), "filename": "selected_features.csv"},
    "parquet": {"directory": str(PATH_CFG.features.root), "filename": "selected_features.parquet"},
    "excel":   {"directory": str(PATH_CFG.features.root), "filename": "selected_features.xlsx"},
}

# ─── Legacy CONFIG dict — keeps old callers working without changes ──────────
CONFIG: dict[str, Any] = {
    "project": {
        "name":    "helioforge-ai",
        "version": "1.0.0",
    },
    "paths": {k: str(v) for k, v in _PATH_MAP.items()},
    "files": {k: str(v) for k, v in _FILE_MAP.items()},
    "exports": _EXPORT_MAP,
    "logging": {"level": "INFO"},
}

PROJECT_ROOT = _REPO_ROOT


def get_path(*keys: str) -> Path:
    """
    Return an absolute Path for a named key.

    Examples
    --------
        get_path("solexs")      ->  /opt/helioforge/raw/solexs
        get_path("features")    ->  /opt/helioforge/features
        get_path("checkpoints") ->  /opt/helioforge/experiments/baseline_tcn/checkpoints
    """
    if len(keys) == 1:
        key = keys[0]
        if key in _PATH_MAP:
            return _PATH_MAP[key]
        # fall back to nested CONFIG["paths"] lookup
        raw = CONFIG["paths"].get(key)
        if raw is not None:
            return Path(raw)
        raise KeyError(f"[config] Unknown path key: '{key}'")

    # nested traversal for multi-key calls
    current: Any = CONFIG["paths"]
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"[config] Unknown config path: '{'/'.join(keys)}'")
        current = current[key]
    return Path(current)


if __name__ == "__main__":
    print("=" * 60)
    print("HelioForge-AI  |  Path Configuration")
    print("=" * 60)
    print(f"  YAML source   : {_PATHS_YAML}")
    print(f"  Dataset root  : {PATH_CFG.dataset_root}")
    print(f"  SoLEXS raw    : {PATH_CFG.raw.solexs}")
    print(f"  HEL1OS raw    : {PATH_CFG.raw.hel1os}")
    print(f"  Processed     : {PATH_CFG.preprocessing.processed}")
    print(f"  Features      : {PATH_CFG.features.root}")
    print(f"  Windows       : {PATH_CFG.windows.root}")
    print(f"  Checkpoints   : {PATH_CFG.experiments.baseline_tcn.checkpoints}")
    print(f"  Logs          : {PATH_CFG.logs.root}")
