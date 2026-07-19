"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
dataset_builder.py

Build an ML-ready dataset by extracting
features from all processed observations.
==========================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.feature_pipeline import (
    FeaturePipeline,
)


class DatasetBuilder:
    """
    Builds an ML dataset from processed observations.
    """

    def __init__(self):

        self.pipeline = FeaturePipeline()

        self.rows: list[dict] = []

    def add_sample(
        self,
        soft_signal,
        hard_signal,
        timestamps,
        solexs_id: str,
        hel1os_id: str,
        label=None,
    ) -> None:

        features = self.pipeline.run(
            soft_signal,
            hard_signal,
            timestamps,
        )
        
        row = {
            "solexs_observation_id": solexs_id,
            "hel1os_observation_id": hel1os_id,
        }

        if label is not None:
            row["label"] = label

        row.update(features)

        self.rows.append(row)

    def build(self) -> pd.DataFrame:

        return pd.DataFrame(self.rows)

    def clear(self):

        self.rows.clear()