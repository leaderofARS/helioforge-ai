"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
missing_values.py

Generate missing-value reports.
==========================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import CONFIG, get_path


class MissingValues:

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
        dataframe: pd.DataFrame,
    ) -> None:

        ##################################################
        # COMPUTE
        ##################################################

        missing = dataframe.isnull().sum()

        report = pd.DataFrame({

            "Feature": missing.index,

            "Missing Values": missing.values,

            "Missing Percentage":
                (missing.values / len(dataframe)) * 100,

        })

        ##################################################
        # SAVE CSV
        ##################################################

        report.to_csv(

            self.output_dir /
            Path(CONFIG["files"]["missing_values_csv"]).name,

            index=False,

        )

        ##################################################
        # PLOT
        ##################################################

        plt.figure(
            figsize=(12,6)
        )

        plt.bar(

            report["Feature"],

            report["Missing Values"],

        )

        plt.xticks(

            rotation=90,

            fontsize=7,

        )

        plt.ylabel(
            "Missing Values"
        )

        plt.title(
            "Missing Value Report",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        plt.savefig(

            self.output_dir /
            Path(CONFIG["files"]["missing_values_png"]).name,

            dpi=300,

        )

        plt.close()

        ##################################################

        print()

        print("="*50)

        print("Missing Value Report Generated")

        print("="*50)

        print(
            report["Missing Values"].sum(),
            "missing values found.",
        )

        print("="*50)