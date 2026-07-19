"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
variance_filter.py

Remove low-variance features.
==========================================================
"""

from __future__ import annotations

import pandas as pd

from sklearn.feature_selection import VarianceThreshold


class VarianceFilter:
    """
    Remove features with variance below
    a specified threshold.
    """

    def __init__(
        self,
        threshold: float = 0.01,
    ) -> None:

        self.threshold = threshold

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        metadata = [
            "solexs_observation_id",
            "hel1os_observation_id",
            "label",
        ]

        metadata_columns = [
            c for c in metadata
            if c in dataframe.columns
        ]

        metadata_df = dataframe[
            metadata_columns
        ].copy()

        feature_df = dataframe.drop(
            columns=metadata_columns,
        )

        selector = VarianceThreshold(
            threshold=self.threshold,
        )

        transformed = selector.fit_transform(
            feature_df,
        )

        selected_columns = feature_df.columns[
            selector.get_support()
        ]

        filtered = pd.DataFrame(
            transformed,
            columns=selected_columns,
            index=dataframe.index,
        )

        return pd.concat(
            [
                metadata_df,
                filtered,
            ],
            axis=1,
        )