"""
scripts/evaluate.py
────────────────────
Stage 4 — Evaluation & Visualisation

Loads processed observations, builds features, runs feature selection,
then generates all visualisation plots and saves them to
/opt/helioforge/reports/figures/.

Run:
    python scripts/evaluate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.feature_selector import FeatureSelector
from src.features.feature_pipeline import FeaturePipeline
from src.pipeline.ingestion.observation_loader import ObservationLoader
from src.utils.config import PATH_CFG

# Visualisation pipeline lives under visualizations/ at repo root
sys.path.insert(0, str(REPO_ROOT / "visualizations"))
from visualization_pipeline import VisualizationPipeline  # noqa: E402


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    return logging.getLogger("helioforge.evaluate")


def main() -> int:
    logger = configure_logging()

    print("=" * 60)
    print("HELIO-FORGE EVALUATION ENTRY POINT")
    print("=" * 60)

    try:
        processed_directory = PATH_CFG.preprocessing.processed
        logger.info("Loading observations for evaluation")
        print(f"[STAGE] Loading processed observations from {processed_directory}")
        loader = ObservationLoader(processed_directory)

        logger.info("Building feature dataset for evaluation")
        print("[STAGE] Extracting features for evaluation dataset")
        pipeline = FeaturePipeline()
        rows = []

        for index, observation in enumerate(loader.load_all(), start=1):
            print(
                f"[STAGE] Preparing evaluation sample {index}: "
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

        logger.info("Running feature selection for evaluation")
        print("[STAGE] Computing feature selection summary")
        selector = FeatureSelector()
        selected_dataframe, _, original_features, variance_features, correlation_features = selector.run(
            dataframe,
            use_feature_importance=False,
        )

        logger.info("Generating evaluation visualizations")
        print(f"[STAGE] Running visualization pipeline → {PATH_CFG.reports.figures}")
        vis_pipeline = VisualizationPipeline()
        vis_pipeline.run(
            dataframe,
            selected_dataframe,
            original_features,
            variance_features,
            correlation_features,
        )

        print("[SUCCESS] Evaluation workflow completed successfully")
        return 0

    except Exception as exc:
        print(f"[ERROR] Evaluation workflow failed: {exc}", file=sys.stderr)
        logger.exception("Evaluation workflow failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
