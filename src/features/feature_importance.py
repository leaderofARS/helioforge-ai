"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

feature_importance.py

Compute feature importance using Random Forest.
==========================================================
"""

from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class FeatureImportance:
    """
    Compute Random Forest feature importance.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        random_state: int = 42,
    ) -> None:

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def compute(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        metadata = [
            "solexs_observation_id",
            "hel1os_observation_id",
        ]

        drop_columns = [
            c for c in metadata
            if c in dataframe.columns
        ]

        X = dataframe.drop(
            columns=drop_columns + ["label"]
        )

        y = dataframe["label"]

        if y.nunique() < 2:
            raise ValueError(
                "Feature importance requires at least "
                "two label classes."
            )

        self.model.fit(X, y)

        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": self.model.feature_importances_,
            }
        )

        importance = importance.sort_values(
            by="importance",
            ascending=False,
        ).reset_index(drop=True)

        return importance