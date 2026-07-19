"""
scripts/features.py
────────────────────
Stage 3 — Feature Engineering

Loads processed observations, runs every feature extractor,
applies variance + correlation filters, then exports the
selected feature dataset to /opt/helioforge/features/.

Run:
    python scripts/features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.feature_pipeline import FeaturePipeline
from src.features.feature_selector import FeatureSelector
from src.pipeline.ingestion.dataset_exporter import DatasetExporter
from src.pipeline.ingestion.observation_loader import ObservationLoader
from src.utils.config import CONFIG, PATH_CFG


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, str(CONFIG["logging"]["level"]).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    return logging.getLogger("helioforge.features")


def main() -> int:
    logger = configure_logging()

    print("=" * 60)
    print("HELIO-FORGE FEATURE ENGINEERING ENTRY POINT")
    print("=" * 60)

    try:
        processed_directory = PATH_CFG.preprocessing.processed
        logger.info("Loading observations for feature engineering")
        print(f"[STAGE] Loading processed observations from {processed_directory}")
        loader = ObservationLoader(processed_directory)

        logger.info("Building feature dataset")
        print("[STAGE] Extracting features for each observation")
        pipeline = FeaturePipeline()
        rows = []

        for index, observation in enumerate(loader.load_all(), start=1):
            print(
                f"[STAGE] Extracting features for observation {index}: "
                f"{observation['solexs_id']} / {observation['hel1os_id']}"
            )
            features = pipeline.run(
                observation["soft_signal"],
                observation["hard_signal"],
                observation["timestamps"],
            )
            row = {
                "solexs_observation_id": observation["solexs_id"],
                "hel1os_observation_id": observation["hel1os_id"],
            }
            row.update(features)
            rows.append(row)

        if not rows:
            raise RuntimeError("No observations were loaded from the processed directories")

        dataframe = pd.DataFrame(rows)

        ##################################################
        # FEATURE SELECTION
        ##################################################

        logger.info("Running feature selection pipeline")
        print("[STAGE] Applying variance and correlation filters")
        selector = FeatureSelector()
        selected_dataframe, _, original_features, variance_features, correlation_features = selector.run(
            dataframe,
            use_feature_importance=False,
        )

        ##################################################
        # EXPORT
        ##################################################

        logger.info("Exporting selected features")
        print(f"[STAGE] Exporting selected features to {PATH_CFG.features.root}")

        exporter = DatasetExporter(PATH_CFG.features.root)
        exporter.export_all(selected_dataframe)

        print(f"[INFO] Original features      : {original_features}")
        print(f"[INFO] After variance filter  : {variance_features}")
        print(f"[INFO] After correlation filter: {correlation_features}")
        print("[SUCCESS] Feature engineering workflow completed successfully")
        return 0

    except Exception as exc:
        print(f"[ERROR] Feature engineering workflow failed: {exc}", file=sys.stderr)
        logger.exception("Feature engineering workflow failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
