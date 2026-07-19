"""
scripts/ingest.py
──────────────────
Stage 3 — Ingestion (Processed → ML Dataset)

Loads processed observations from /opt/helioforge/preprocessing/processed,
extracts features via FeaturePipeline, and exports the ML-ready dataset
to /opt/helioforge/features/.

Run:
    python scripts/ingest.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.ingestion.dataset_builder import DatasetBuilder
from src.pipeline.ingestion.dataset_exporter import DatasetExporter
from src.pipeline.ingestion.observation_loader import ObservationLoader
from src.utils.config import CONFIG, PATH_CFG


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, str(CONFIG["logging"]["level"]).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    return logging.getLogger("helioforge.ingest")


def main() -> int:
    logger = configure_logging()

    print("=" * 60)
    print("HELIO-FORGE INGESTION ENTRY POINT")
    print("=" * 60)

    try:
        processed_directory = PATH_CFG.preprocessing.processed
        logger.info("Loading processed observations from %s", processed_directory)
        print(f"[STAGE] Loading processed observations from {processed_directory}")
        loader = ObservationLoader(processed_directory)

        logger.info("Building dataset from observations")
        print("[STAGE] Building ML-ready dataset")
        builder = DatasetBuilder()

        for index, observation in enumerate(loader.load_all(), start=1):
            print(
                f"[STAGE] Processing observation {index}: "
                f"{observation['solexs_id']} / {observation['hel1os_id']}"
            )
            builder.add_sample(
                observation["soft_signal"],
                observation["hard_signal"],
                observation["timestamps"],
                observation["solexs_id"],
                observation["hel1os_id"],
            )

        if not builder.rows:
            raise RuntimeError("No observations were loaded from the processed directories")

        dataset = builder.build()

        logger.info("Exporting dataset to configured output formats")
        print("[STAGE] Exporting dataset")

        # All three formats go to the same features root directory
        exporter = DatasetExporter(PATH_CFG.features.root)
        exporter.export_all(dataset)

        print("[SUCCESS] Ingestion workflow completed successfully")
        return 0

    except Exception as exc:
        print(f"[ERROR] Ingestion workflow failed: {exc}", file=sys.stderr)
        logger.exception("Ingestion workflow failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
