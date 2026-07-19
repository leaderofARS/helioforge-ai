"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
visualization_pipeline.py

Master visualization pipeline.
==========================================================
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.feature_count import (
    FeatureCountPlot,
)

from src.evaluation.correlation_heatmap import (
    CorrelationHeatmap,
)

from src.evaluation.feature_distribution import (
    FeatureDistribution,
)

from src.evaluation.missing_values import (
    MissingValues,
)

from src.evaluation.feature_selection_summary import (
    FeatureSelectionSummary,
)

from src.evaluation.pca_analysis import (
    PCAAnalysis,
)


class VisualizationPipeline:
    """
    Master visualization pipeline.
    """

    def __init__(self) -> None:

        self.feature_count = FeatureCountPlot()

        self.heatmap = CorrelationHeatmap()

        self.distribution = FeatureDistribution()

        self.missing = MissingValues()

        self.summary = FeatureSelectionSummary()

        self.pca = PCAAnalysis()

    ##################################################

    def run(

        self,

        dataframe: pd.DataFrame,

        selected_dataframe: pd.DataFrame,

        original_features: int,

        variance_features: int,

        correlation_features: int,

    ) -> None:

        print()

        print("=" * 60)

        print("HELIO-FORGE VISUALIZATION PIPELINE")

        print("=" * 60)

        ##################################################
        # FEATURE COUNT
        ##################################################

        self.feature_count.plot(

            original_features,

            correlation_features,

        )

        ##################################################
        # CORRELATION HEATMAPS
        ##################################################

        self.heatmap.run(
            dataframe
        )

        ##################################################
        # FEATURE DISTRIBUTIONS
        ##################################################

        self.distribution.run(
            selected_dataframe
        )

        ##################################################
        # MISSING VALUES
        ##################################################

        self.missing.run(
            dataframe
        )

        ##################################################
        # FEATURE SUMMARY
        ##################################################

        self.summary.run(

            original_features,

            variance_features,

            correlation_features,

        )

        ##################################################
        # PCA
        ##################################################

        self.pca.run(
            dataframe
        )

        ##################################################

        print()

        print("=" * 60)

        print("Visualization Pipeline Complete")

        print("=" * 60)