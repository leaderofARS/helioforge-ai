"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
feature_selector.py

Master feature selection pipeline.
==========================================================
"""

from __future__ import annotations

import pandas as pd

from src.features.variance_filter import (
    VarianceFilter,
)

from src.features.correlation_filter import (
    CorrelationFilter,
)

from src.features.feature_importance import (
    FeatureImportance,
)


class FeatureSelector:
    """
    Complete feature selection pipeline.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
    ) -> None:

        self.variance = VarianceFilter(
            threshold=variance_threshold,
        )

        self.correlation = CorrelationFilter(
            threshold=correlation_threshold,
        )

        self.importance = FeatureImportance()

    ##################################################

    @staticmethod
    def _feature_count(
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Count engineered features
        (excluding metadata columns).
        """

        metadata_columns = {

            "solexs_observation_id",

            "hel1os_observation_id",

            "label",

        }

        return len(

            [

                column

                for column in dataframe.columns

                if column not in metadata_columns

            ]

        )

    ##################################################

    def run(

        self,

        dataframe: pd.DataFrame,

        use_feature_importance: bool = False,

    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame | None,
        int,
        int,
        int,
    ]:

        print()

        print("=" * 60)
        print("FEATURE SELECTION PIPELINE")
        print("=" * 60)

        ##################################################
        # ORIGINAL
        ##################################################

        original_features = self._feature_count(
            dataframe
        )

        print(
            f"Original Features : {original_features}"
        )

        ##################################################
        # VARIANCE FILTER
        ##################################################

        dataframe = self.variance.fit_transform(
            dataframe
        )

        variance_features = self._feature_count(
            dataframe
        )

        print(
            f"After Variance Filter : {variance_features}"
        )

        ##################################################
        # CORRELATION FILTER
        ##################################################

        dataframe = self.correlation.fit_transform(
            dataframe
        )

        correlation_features = self._feature_count(
            dataframe
        )

        print(
            f"After Correlation Filter : {correlation_features}"
        )

        ##################################################
        # FEATURE IMPORTANCE
        ##################################################

        importance = None

        if use_feature_importance:

            print()

            print(
                "Running Feature Importance..."
            )

            importance = self.importance.compute(
                dataframe
            )

            print(
                "Feature Importance Complete"
            )

        ##################################################

        print()

        print("=" * 60)
        print("FEATURE SELECTION COMPLETE")
        print("=" * 60)

        return (

            dataframe,

            importance,

            original_features,

            variance_features,

            correlation_features,

        )