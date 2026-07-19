"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
pca_analysis.py

Principal Component Analysis visualization.
==========================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import CONFIG, get_path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAnalysis:

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
        # REMOVE CONSTANT FEATURES
        ##################################################

        dataframe = dataframe.loc[
            :,
            dataframe.nunique() > 1,
        ]

        ##################################################
        # SCALE
        ##################################################

        scaler = StandardScaler()

        X = scaler.fit_transform(
            dataframe
        )

        ##################################################
        # PCA
        ##################################################

        pca = PCA()

        transformed = pca.fit_transform(
            X
        )

        explained = pca.explained_variance_ratio_

        cumulative = explained.cumsum()

        ##################################################
        # SAVE SUMMARY
        ##################################################

        summary = pd.DataFrame({

            "Principal Component":
                range(
                    1,
                    len(explained)+1,
                ),

            "Explained Variance":
                explained,

            "Cumulative Variance":
                cumulative,

        })

        pca_csv_path = self.output_dir / Path(CONFIG["files"]["pca_summary_csv"]).name
        summary.to_csv(

            pca_csv_path,

            index=False,

        )

        ##################################################
        # EXPLAINED VARIANCE
        ##################################################

        plt.figure(
            figsize=(10,6)
        )

        plt.bar(

            range(
                1,
                len(explained)+1,
            ),

            explained,

        )

        plt.xlabel(
            "Principal Component"
        )

        plt.ylabel(
            "Explained Variance Ratio"
        )

        plt.title(
            "PCA Explained Variance",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        explained_png_path = self.output_dir / Path(CONFIG["files"]["pca_explained_variance_png"]).name
        plt.savefig(

            explained_png_path,

            dpi=300,

        )

        plt.close()

        ##################################################
        # CUMULATIVE VARIANCE
        ##################################################

        plt.figure(
            figsize=(10,6)
        )

        plt.plot(

            range(
                1,
                len(cumulative)+1,
            ),

            cumulative,

            marker="o",

        )

        plt.xlabel(
            "Principal Component"
        )

        plt.ylabel(
            "Cumulative Explained Variance"
        )

        plt.grid(True)

        plt.title(
            "Cumulative PCA Variance",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        cumulative_png_path = self.output_dir / Path(CONFIG["files"]["pca_cumulative_variance_png"]).name
        plt.savefig(

            cumulative_png_path,

            dpi=300,

        )

        plt.close()

        ##################################################
        # PCA PROJECTION
        ##################################################

        if transformed.shape[1] >= 2:

            plt.figure(
                figsize=(8,8)
            )

            plt.scatter(

                transformed[:,0],

                transformed[:,1],

                s=60,

            )

            plt.xlabel("PC1")

            plt.ylabel("PC2")

            plt.title(
                "PCA Projection",
                fontsize=16,
                fontweight="bold",
            )

            plt.tight_layout()

            projection_png_path = self.output_dir / Path(CONFIG["files"]["pca_projection_png"]).name
            plt.savefig(

                projection_png_path,

                dpi=300,

            )

            plt.close()

        ##################################################

        print()

        print("="*50)

        print("PCA Analysis Complete")

        print("="*50)

        print(summary)

        print("="*50)