"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

correlation_filter.py

Remove highly correlated features.
==========================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class CorrelationFilter:
    """
    Remove highly correlated features.
    """

    def __init__(
        self,
        threshold: float = 0.95,
    ) -> None:

        self.threshold = threshold

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        ##################################################
        # METADATA
        ##################################################

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

        ##################################################
        # CORRELATION MATRIX
        ##################################################

        correlation_matrix = feature_df.corr().abs()

        upper = correlation_matrix.where(
            np.triu(
                np.ones(correlation_matrix.shape),
                k=1,
            ).astype(bool)
        )

        ##################################################
        # FIND FEATURES TO DROP
        ##################################################

        drop_columns = [
            column
            for column in upper.columns
            if any(
                upper[column] > self.threshold
            )
        ]

        ##################################################
        # DROP FEATURES
        ##################################################

        filtered = feature_df.drop(
            columns=drop_columns,
        )

        ##################################################
        # RETURN
        ##################################################

        return pd.concat(
            [
                metadata_df,
                filtered,
            ],
            axis=1,
        )