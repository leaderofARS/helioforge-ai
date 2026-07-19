from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.feature_selector import FeatureSelector
from src.ingestion.dataset_builder import DatasetBuilder
from src.ingestion.dataset_exporter import DatasetExporter
from src.ingestion.observation_loader import ObservationLoader
from src.utils.config import CONFIG, get_path


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
        processed_directory = get_path("processed")
        logger.info("Loading observations for feature engineering")
        print("[STAGE] Loading processed observations")
        loader = ObservationLoader(processed_directory)

        logger.info("Building feature dataset")
        print("[STAGE] Extracting features for each observation")
        builder = DatasetBuilder()

        for index, observation in enumerate(loader.load_all(), start=1):
            print(f"[STAGE] Extracting features for observation {index}: {observation['solexs_id']} / {observation['hel1os_id']}")
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

        logger.info("Running feature selection pipeline")
        print("[STAGE] Applying variance and correlation filters")
        selector = FeatureSelector()
        selected_dataframe, _, original_features, variance_features, correlation_features = selector.run(
            dataframe,
            use_feature_importance=False,
        )

        logger.info("Exporting selected features")
        print("[STAGE] Exporting selected features")

        selected_name = f"selected_{CONFIG['exports']['csv']['filename']}"
        selected_parquet_name = f"selected_{CONFIG['exports']['parquet']['filename']}"
        selected_excel_name = f"selected_{CONFIG['exports']['excel']['filename']}"

        csv_exporter = DatasetExporter(Path(CONFIG["exports"]["csv"]["directory"]))
        csv_exporter.export_csv(selected_dataframe, filename=selected_name)

        parquet_exporter = DatasetExporter(Path(CONFIG["exports"]["parquet"]["directory"]))
        parquet_exporter.export_parquet(selected_dataframe, filename=selected_parquet_name)

        excel_exporter = DatasetExporter(Path(CONFIG["exports"]["excel"]["directory"]))
        excel_exporter.export_excel(selected_dataframe, filename=selected_excel_name)

        print(f"[INFO] Original features: {original_features}")
        print(f"[INFO] After variance filter: {variance_features}")
        print(f"[INFO] After correlation filter: {correlation_features}")
        print("[SUCCESS] Feature engineering workflow completed successfully")
        return 0

    except Exception as exc:
        print(f"[ERROR] Feature engineering workflow failed: {exc}", file=sys.stderr)
        logger.exception("Feature engineering workflow failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
