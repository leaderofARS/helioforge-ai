from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.visualization_pipeline import VisualizationPipeline
from src.features.feature_selector import FeatureSelector
from src.ingestion.dataset_builder import DatasetBuilder
from src.ingestion.observation_loader import ObservationLoader
from src.utils.config import get_path


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
        processed_directory = get_path("processed")
        logger.info("Loading observations for evaluation")
        print("[STAGE] Loading processed observations")
        loader = ObservationLoader(processed_directory)

        logger.info("Building feature dataset for evaluation")
        print("[STAGE] Building feature dataset")
        builder = DatasetBuilder()

        for index, observation in enumerate(loader.load_all(), start=1):
            print(f"[STAGE] Preparing evaluation sample {index}: {observation['solexs_id']} / {observation['hel1os_id']}")
            builder.add_sample(
                observation["soft_signal"],
                observation["hard_signal"],
                observation["timestamps"],
                observation["solexs_id"],
                observation["hel1os_id"],
            )

        if not builder.rows:
            raise RuntimeError("No observations were loaded from the processed directories")

        dataframe = builder.build()

        logger.info("Running feature selection for evaluation")
        print("[STAGE] Computing feature selection summary")
        selector = FeatureSelector()
        selected_dataframe, _, original_features, variance_features, correlation_features = selector.run(
            dataframe,
            use_feature_importance=False,
        )

        logger.info("Generating evaluation visualizations")
        print("[STAGE] Running visualization pipeline")
        pipeline = VisualizationPipeline()
        pipeline.run(
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
