"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
feature_selection_summary.py

Generate feature selection summary.
==========================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import CONFIG, get_path


class FeatureSelectionSummary:

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

    def run(

        self,

        original_features: int,

        variance_features: int,

        correlation_features: int,

    ) -> None:

        ##################################################
        # SUMMARY TABLE
        ##################################################

        summary = pd.DataFrame({

            "Stage": [

                "Original",

                "After Variance Filter",

                "After Correlation Filter",

            ],

            "Features": [

                original_features,

                variance_features,

                correlation_features,

            ],

        })

        ##################################################
        # SAVE CSV
        ##################################################

        summary.to_csv(

            self.output_dir /
            Path(CONFIG["files"]["feature_selection_summary_csv"]).name,

            index=False,

        )

        ##################################################
        # BAR CHART
        ##################################################

        plt.figure(

            figsize=(8,6)

        )

        plt.bar(

            summary["Stage"],

            summary["Features"],

        )

        ##################################################

        for index, value in enumerate(

            summary["Features"]

        ):

            plt.text(

                index,

                value + 1,

                str(value),

                ha="center",

                fontsize=11,

                fontweight="bold",

            )

        ##################################################

        plt.ylabel(

            "Number of Features"

        )

        plt.title(

            "Feature Selection Summary",

            fontsize=16,

            fontweight="bold",

        )

        plt.tight_layout()

        plt.savefig(

            self.output_dir /

            Path(CONFIG["files"]["feature_selection_summary_png"]).name,

            dpi=300,

        )

        plt.close()

        ##################################################

        print()

        print("="*50)

        print("Feature Selection Summary Generated")

        print("="*50)

        print(summary)

        print("="*50)