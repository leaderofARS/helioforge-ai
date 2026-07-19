"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
feature_count.py

Visualize feature counts before and after
feature selection.
==========================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.config import CONFIG, get_path


class FeatureCountPlot:

    def __init__(
        self,
        output_dir: str | Path | None = None,
    ) -> None:

        self.output_dir = Path(output_dir) if output_dir is not None else get_path("visualizations")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot(
        self,
        original_features: int,
        selected_features: int,
    ) -> None:

        plt.figure(figsize=(6, 5))

        plt.bar(
            ["Original", "Selected"],
            [
                original_features,
                selected_features,
            ],
        )

        plt.ylabel("Number of Features")

        plt.title("Feature Count")

        plt.tight_layout()

        plt.savefig(
            self.output_dir
            / Path(CONFIG["files"]["feature_count_png"]).name,
            dpi=300,
        )

        plt.close()