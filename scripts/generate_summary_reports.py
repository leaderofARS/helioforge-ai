"""
scripts/generate_summary_reports.py
───────────────────────────────────
Generates comprehensive quantitative CSV summary reports for:
1. Feature Selection & Filtering Reports (variance_threshold_report.csv, correlation_removal_report.csv, selected_feature_names.csv)
2. Evaluation Reports (pca_summary.csv, correlation_matrix.csv, top_correlations.csv)
3. Window Generation Statistics (window_generation_summary.csv, window_metadata.csv)

Usage
-----
    python scripts/generate_summary_reports.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.feature_selector import FeatureSelector
from src.pipeline.ingestion.observation_loader import ObservationLoader
from src.utils.config import PATH_CFG

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("helioforge.reports")


def generate_feature_selection_reports(df: pd.DataFrame, tables_dir: Path, features_dir: Path) -> None:
    logger.info("Generating feature selection reports...")

    # Identify numeric feature columns (exclude non-feature IDs if present)
    ignore_cols = {"observation_id", "solexs_id", "hel1os_id", "time", "timestamp"}
    feature_cols = [c for c in df.columns if c.lower() not in ignore_cols]
    feature_df = df[feature_cols].select_dtypes(include=[np.number])

    # 1. Variance Threshold Report
    variances = feature_df.var()
    var_threshold = 1e-4
    var_report = pd.DataFrame(
        {
            "feature_name": variances.index,
            "variance": variances.values,
            "threshold": var_threshold,
            "passed_variance_filter": variances.values >= var_threshold,
        }
    ).sort_values(by="variance", ascending=False)

    passed_var_cols = var_report[var_report["passed_variance_filter"]]["feature_name"].tolist()

    # 2. Correlation Removal Report
    filtered_df = feature_df[passed_var_cols]
    corr_matrix = filtered_df.corr().abs()
    corr_threshold = 0.85

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    corr_records = []

    for col in upper.columns:
        high_corr_partners = upper.index[upper[col] > corr_threshold].tolist()
        for partner in high_corr_partners:
            r_val = float(upper.loc[partner, col])
            status = "dropped" if col not in to_drop else "already_dropped"
            corr_records.append(
                {
                    "feature": col,
                    "correlated_with": partner,
                    "absolute_correlation": r_val,
                    "threshold": corr_threshold,
                    "action": status,
                }
            )
            if col not in to_drop:
                to_drop.append(col)

    corr_report = pd.DataFrame(corr_records)
    selected_cols = [c for c in passed_var_cols if c not in to_drop]

    # 3. Selected Feature Names Report
    selected_report = pd.DataFrame(
        {
            "feature_index": range(1, len(selected_cols) + 1),
            "feature_name": selected_cols,
            "variance": [variances[c] for c in selected_cols],
            "group": [
                "soft_xray" if "soft" in c.lower() or "solexs" in c.lower()
                else "hard_xray" if "hard" in c.lower() or "hel1os" in c.lower()
                else "spectral" if "spec" in c.lower() or "fft" in c.lower()
                else "wavelet" if "wavelet" in c.lower() or "db4" in c.lower()
                else "statistical"
                for c in selected_cols
            ],
        }
    )

    # Save outputs
    var_report.to_csv(tables_dir / "variance_threshold_report.csv", index=False)
    var_report.to_csv(features_dir / "variance_threshold_report.csv", index=False)

    if not corr_report.empty:
        corr_report.to_csv(tables_dir / "correlation_removal_report.csv", index=False)
        corr_report.to_csv(features_dir / "correlation_removal_report.csv", index=False)

    selected_report.to_csv(tables_dir / "selected_feature_names.csv", index=False)
    selected_report.to_csv(features_dir / "selected_feature_names.csv", index=False)

    logger.info("Feature selection reports saved successfully.")


def generate_evaluation_reports(df: pd.DataFrame, tables_dir: Path, features_dir: Path) -> None:
    logger.info("Generating PCA and Correlation evaluation reports...")

    ignore_cols = {"observation_id", "solexs_id", "hel1os_id", "time", "timestamp"}
    feature_cols = [c for c in df.columns if c.lower() not in ignore_cols]
    numeric_df = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)

    # 1. Full Correlation Matrix
    corr_df = numeric_df.corr()
    corr_df.to_csv(tables_dir / "correlation_matrix.csv")
    corr_df.to_csv(features_dir / "correlation_matrix.csv")

    # 2. Top Correlated Feature Pairs
    upper_tri = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    stacked_corr = upper_tri.stack().reset_index()
    stacked_corr.columns = ["feature_1", "feature_2", "correlation"]
    stacked_corr["abs_correlation"] = stacked_corr["correlation"].abs()
    top_corr = stacked_corr.sort_values(by="abs_correlation", ascending=False).drop(
        columns=["abs_correlation"]
    )

    top_corr.to_csv(tables_dir / "top_correlations.csv", index=False)
    top_corr.to_csv(features_dir / "top_correlations.csv", index=False)

    # 3. PCA Summary Report
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    pca = PCA()
    pca.fit(scaled_data)

    pca_summary = pd.DataFrame(
        {
            "principal_component": [f"PC_{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "singular_values": pca.singular_values_,
        }
    )

    pca_summary.to_csv(tables_dir / "pca_summary.csv", index=False)
    pca_summary.to_csv(features_dir / "pca_summary.csv", index=False)

    logger.info("Evaluation reports (PCA & Correlation) saved successfully.")


def generate_window_reports(windows_dir: Path, tables_dir: Path) -> None:
    logger.info("Generating window statistics reports...")

    splits = ["train", "val", "test"]
    summary_records = []

    for split in splits:
        file_path = windows_dir / f"{split}.pt"
        if not file_path.exists():
            logger.warning(f"Window file not found: {file_path}")
            continue

        data_dict = torch.load(file_path, weights_only=False)
        tensor = data_dict["sequences"]

        if isinstance(tensor, torch.Tensor):
            n_samples, n_channels, win_len = tensor.shape
            tensor_np = tensor.numpy()

            summary_records.append(
                {
                    "split": split,
                    "filename": f"{split}.pt",
                    "num_windows": n_samples,
                    "num_channels": n_channels,
                    "window_length_steps": win_len,
                    "window_duration_sec": win_len,
                    "total_data_points": n_samples * n_channels * win_len,
                    "min_value": float(np.min(tensor_np)),
                    "max_value": float(np.max(tensor_np)),
                    "mean_value": float(np.mean(tensor_np)),
                    "std_value": float(np.std(tensor_np)),
                    "nan_count": int(np.isnan(tensor_np).sum()),
                }
            )

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(windows_dir / "window_generation_summary.csv", index=False)
    summary_df.to_csv(tables_dir / "window_generation_summary.csv", index=False)

    # Window Metadata Report per Observation
    loader = ObservationLoader(PATH_CFG.preprocessing.processed)
    obs_records = []
    window_size = 512
    stride = 32

    for idx, obs in enumerate(loader.load_all(), start=1):
        soft_len = len(obs["soft_signal"])
        hard_len = len(obs["hard_signal"])
        min_len = min(soft_len, hard_len)
        n_windows = max(0, (min_len - window_size) // stride + 1) if min_len >= window_size else 0

        obs_records.append(
            {
                "observation_index": idx,
                "solexs_id": obs["solexs_id"],
                "hel1os_id": obs["hel1os_id"],
                "soft_signal_length": soft_len,
                "hard_signal_length": hard_len,
                "synchronized_length": min_len,
                "generated_windows": n_windows,
                "window_size": window_size,
                "stride": stride,
            }
        )

    obs_metadata_df = pd.DataFrame(obs_records)
    obs_metadata_df.to_csv(windows_dir / "window_metadata.csv", index=False)
    obs_metadata_df.to_csv(tables_dir / "window_metadata.csv", index=False)

    logger.info("Window statistics reports saved successfully.")


def main() -> int:
    print("=" * 60)
    print("HELIO-FORGE AI  |  GENERATE SUMMARY REPORTS & METRICS")
    print("=" * 60)

    tables_dir = PATH_CFG.reports.root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    features_dir = PATH_CFG.features.root
    windows_dir = PATH_CFG.windows.root

    features_parquet = PATH_CFG.features.parquet
    features_csv = PATH_CFG.features.csv

    if features_parquet.exists():
        df = pd.read_parquet(features_parquet)
    elif features_csv.exists():
        df = pd.read_csv(features_csv)
    else:
        logger.error(f"No feature matrix found at {features_parquet} or {features_csv}")
        return 1

    generate_feature_selection_reports(df, tables_dir, features_dir)
    generate_evaluation_reports(df, tables_dir, features_dir)
    generate_window_reports(windows_dir, tables_dir)

    print("\n" + "=" * 60)
    print("  [SUCCESS] All Summary CSV Reports Generated Successfully!")
    print("  Tables Saved To  : ", tables_dir)
    print("  Features Saved To: ", features_dir)
    print("  Windows Saved To : ", windows_dir)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
