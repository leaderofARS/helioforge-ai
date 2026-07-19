"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
feature_distribution.py

Adaptive feature distribution visualization.
==========================================================
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import get_path


class FeatureDistribution:

    ##################################################
    # TITLES
    ##################################################

    TITLES = {

        "soft": "Soft X-ray Feature Distributions",

        "hard": "Hard X-ray Feature Distributions",

        "temporal": "Temporal Feature Distributions",

        "frequency": "Frequency Feature Distributions",

        "wavelet": "Wavelet Feature Distributions",

        "entropy": "Entropy Feature Distributions",

        "correlation": "Correlation Feature Distributions",
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

    def __init__(

        self,

        output_dir: str | Path | None = None,

    ) -> None:

        self.output_dir = Path(output_dir) if output_dir is not None else get_path("distributions")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    ##################################################

    def _select_columns(

        self,

        dataframe: pd.DataFrame,

        prefixes: tuple[str, ...],

    ) -> list[str]:

        columns = []

        for column in dataframe.columns:

            for prefix in prefixes:

                if column.startswith(prefix):

                    columns.append(column)

                    break

        return columns

    ##################################################

    def _plot_single(

        self,

        ax,

        values: np.ndarray,

        column: str,

        small_dataset: bool,

    ) -> None:

        ##################################################
        # CONSTANT FEATURE
        ##################################################

        if np.isclose(
            values.max(),
            values.min(),
        ):

            ax.text(
                0.5,
                0.55,
                "Constant Feature",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

            ax.text(
                0.5,
                0.42,
                f"Value = {values[0]:.3f}",
                ha="center",
                va="center",
                fontsize=10,
            )

            ax.set_title(column, fontsize=9)

            ax.set_xticks([])

            ax.set_yticks([])

            return

        ##################################################
        # SMALL DATASET
        ##################################################

        if small_dataset:

            ax.scatter(

                values,

                np.ones_like(values),

                s=35,

            )

            ax.set_ylim(
                0.8,
                1.2,
            )

            ax.set_yticks([])

        ##################################################
        # LARGE DATASET
        ##################################################

        else:

            bins = min(
                30,
                max(
                    10,
                    int(
                        np.sqrt(
                            len(values)
                        )
                    ),
                ),
            )

            ax.hist(
                values,
                bins=bins,
                edgecolor="black",
            )

        ax.set_title(
            column,
            fontsize=9,
        )
        
        ##################################################
    # PLOT GROUP
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

        n = len(columns)

        cols = min(4, n)

        rows = math.ceil(n / cols)

        small_dataset = len(dataframe) < 30

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(
                cols * 4,
                rows * 3.5,
            ),
        )

        ##################################################
        # HANDLE SINGLE AXIS
        ##################################################

        if not isinstance(axes, np.ndarray):

            axes = np.array([axes])

        axes = axes.flatten()

        ##################################################
        # DRAW EACH FEATURE
        ##################################################

        for ax, column in zip(axes, columns):

            values = (
                dataframe[column]
                .dropna()
                .astype(float)
                .to_numpy()
            )

            if values.size == 0:

                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    fontsize=11,
                )

                ax.set_title(
                    column,
                    fontsize=9,
                )

                ax.set_xticks([])
                ax.set_yticks([])

                continue

            self._plot_single(

                ax,

                values,

                column,

                small_dataset,

            )

        ##################################################
        # REMOVE UNUSED AXES
        ##################################################

        for ax in axes[n:]:

            ax.axis("off")

        ##################################################
        # TITLE
        ##################################################

        fig.suptitle(

            self.TITLES[group_name],

            fontsize=18,

            fontweight="bold",

        )

        plt.tight_layout(

            rect=[0, 0, 1, 0.96]

        )

        ##################################################
        # SAVE
        ##################################################

        plt.savefig(

            self.output_dir /

            f"{group_name}_distribution.png",

            dpi=300,

            bbox_inches="tight",

        )

        plt.close()

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
        # SMALL DATASET WARNING
        ##################################################

        if len(dataframe) < 30:

            print()

            print(
                "=" * 50
            )

            print(
                "WARNING : SMALL DATASET"
            )

            print(
                f"Samples : {len(dataframe)}"
            )

            print(
                "Using strip plots instead of histograms."
            )

            print(
                "=" * 50
            )

            print()

        ##################################################
        # GENERATE FIGURES
        ##################################################

        generated = 0

        for group, prefixes in self.FEATURE_GROUPS.items():

            self._plot_group(

                dataframe,

                group,

                prefixes,

            )

            generated += 1

        ##################################################
        # SUMMARY
        ##################################################

        print()

        print("=" * 50)

        print("Feature Distribution Complete")

        print("=" * 50)

        print(f"Groups Generated : {generated}")

        print(f"Output Folder    : {self.output_dir}")

        print("=" * 50)