"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

correlation_heatmap.py

Generate grouped correlation heatmaps
for different feature families.

Author: HelioForge AI
==========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import CONFIG, get_path


class CorrelationHeatmap:
    """
    Generate grouped correlation heatmaps
    and export correlation matrices.
    """

    ##################################################
    # TITLES
    ##################################################

    TITLES = {
        "soft": "Soft X-ray Feature Correlation Matrix",
        "hard": "Hard X-ray Feature Correlation Matrix",
        "temporal": "Temporal Feature Correlation Matrix",
        "frequency": "Frequency-domain Feature Correlation Matrix",
        "wavelet": "Wavelet Feature Correlation Matrix",
        "entropy": "Entropy Feature Correlation Matrix",
        "correlation": "Cross-Signal Correlation Matrix",
    }

    ##################################################
    # FEATURE GROUPS
    ##################################################

    FEATURE_GROUPS = {

        "soft": (
            "soft_",
        ),

        "hard": (
            "hard_",
        ),

        "temporal": (
            "trend_",
            "autocorr_",
            "zero_",
            "peak_",
            "longest_",
            "energy_growth",
            "rolling_",
            "gradient",
            "duration",
            "signal_",
            "coefficient_",
        ),

        "frequency": (
            "dominant_frequency",
            "spectral_",
            "low_high",
            "peak_frequency",
            "spectrum_",
        ),

        "wavelet": (
            "wavelet_",
            "detail_",
            "dominant_wavelet",
        ),

        "entropy": (
            "shannon_",
            "energy_entropy",
            "histogram_",
            "approximate_",
            "entropy_ratio",
        ),

        "correlation": (
            "pearson_",
            "spearman_",
            "maximum_cross",
            "lag_at",
            "covariance",
            "mean_difference",
            "std_difference",
            "rms_difference",
        ),
    }

    ##################################################
    # INIT
    ##################################################

    def __init__(
        self,
        output_dir: str | Path | None = None,
    ) -> None:

        self.output_dir = Path(output_dir) if output_dir is not None else get_path("visualizations")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    ##################################################
    # SELECT COLUMNS
    ##################################################

    def _select_columns(
    self,
    dataframe: pd.DataFrame,
    prefixes: tuple[str, ...],
    ) -> list[str]:

        return [

            column

            for column in dataframe.columns

            if column.startswith(prefixes)

        ]

    ##################################################
    # SINGLE HEATMAP
    ##################################################

    def _plot_group(
        self,
        dataframe: pd.DataFrame,
        group_name: str,
        prefixes: tuple[str, ...],
    ) -> None:

        columns = self._select_columns(
            dataframe,
            prefixes,
        )

        if len(columns) == 0:
            return

        correlation = cast(pd.DataFrame, dataframe[
            columns
        ]).corr()

        ##################################################
        # SAVE CORRELATION MATRIX
        ##################################################

        csv_path = self.output_dir / f"{group_name}_{Path(CONFIG['files']['correlation_matrix_csv']).name}"
        correlation.to_csv(csv_path)

        ##################################################
        # ADAPTIVE FIGURE SIZE
        ##################################################

        n = len(columns)

        width = max(
            8,
            n * 1.2,
        )

        height = max(
            6,
            n * 0.9,
        )

        heatmap_path = self.output_dir / f"{group_name}_{Path(CONFIG['files']['correlation_heatmap_png']).name}"

        if len(columns) == 1:
            plt.figure(figsize=(4, 3))
            plt.text(
                0.5,
                0.5,
                f"Only one feature available for {group_name}",
                ha="center",
                va="center",
                fontsize=10,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                heatmap_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            return

        plt.figure(
            figsize=(width, height),
        )

        image = plt.imshow(
            correlation,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            aspect="equal",
        )

        plt.colorbar(
            image,
            label="Correlation",
        )

        plt.xticks(
            range(n),
            columns,
            rotation=90,
            fontsize=8,
        )

        plt.yticks(
            range(n),
            columns,
            fontsize=8,
        )

        plt.title(
            self.TITLES[group_name],
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        plt.savefig(
            heatmap_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        
    ##################################################
    # TOP CORRELATED PAIRS
    ##################################################

    def _save_top_correlations(
        self,
        dataframe: pd.DataFrame,
        group_name: str,
        prefixes: tuple[str, ...],
        threshold: float = 0.90,
    ) -> None:

        columns = self._select_columns(
            dataframe,
            prefixes,
        )

        top_csv_path = self.output_dir / f"{group_name}_{Path(CONFIG['files']['top_correlations_csv']).name}"

        if len(columns) < 2:
            with open(top_csv_path, "w", encoding="utf-8") as f:
                f.write("Feature 1,Feature 2,Correlation\n")
            return

        correlation = (
            cast(pd.DataFrame, dataframe[columns])
            .corr()
            .abs()
        )

        rows = []

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                value = correlation.iloc[i, j]

                if value >= threshold:

                    rows.append(
                        {
                            "Feature 1": columns[i],
                            "Feature 2": columns[j],
                            "Correlation": value,
                        }
                    )

        if len(rows) == 0:
            with open(top_csv_path, "w", encoding="utf-8") as f:
                f.write("Feature 1,Feature 2,Correlation\n")
            return

        top = (
            pd.DataFrame(rows)
            .sort_values(
                by="Correlation",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        top.to_csv(
            top_csv_path,
            index=False,
        )

    ##################################################
    # RUN
    ##################################################

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        ##################################################
        # REMOVE METADATA
        ##################################################

        metadata = [
            "solexs_observation_id",
            "hel1os_observation_id",
            "label",
        ]

        dataframe = dataframe.drop(
            columns=[
                c
                for c in metadata
                if c in dataframe.columns
            ]
        )

        ##################################################
        # GENERATE EVERYTHING
        ##################################################

        generated = 0

        for group, prefixes in self.FEATURE_GROUPS.items():

            self._plot_group(
                dataframe,
                group,
                prefixes,
            )

            self._save_top_correlations(
                dataframe,
                group,
                prefixes,
            )

            generated += 1

        print("\nVisualization Complete")
        print("-" * 40)
        print(f"Groups Processed : {generated}")
        print(f"Output Folder    : {self.output_dir}")
        print("-" * 40)