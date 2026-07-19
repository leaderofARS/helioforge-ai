from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.hel1os import process_hel1os
from src.preprocessing.solexs import process_solexs
from src.preprocessing.synchronization import save_synchronization_report, synchronize_datasets
from src.utils.config import CONFIG


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, str(CONFIG["logging"]["level"]).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    return logging.getLogger("helioforge.preprocess")


def main() -> int:
    logger = configure_logging()

    print("=" * 60)
    print("HELIO-FORGE PREPROCESSING ENTRY POINT")
    print("=" * 60)

    try:
        logger.info("Starting HEL1OS preprocessing workflow")
        print("[STAGE] Running HEL1OS preprocessing")
        hel1os_data = process_hel1os()

        logger.info("Starting SoLEXS preprocessing workflow")
        print("[STAGE] Running SoLEXS preprocessing")
        solexs_data = process_solexs()

        logger.info("Saving metadata outputs")
        print("[STAGE] Saving generated metadata")

        hel1os_metadata_path = Path(CONFIG["files"]["hel1os_metadata"])
        hel1os_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        hel1os_data["metadata"].to_csv(hel1os_metadata_path, index=False)

        solexs_metadata_path = Path(CONFIG["files"]["solexs_metadata"])
        solexs_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        solexs_data["metadata"].to_csv(solexs_metadata_path, index=False)

        logger.info("Synchronizing observations")
        print("[STAGE] Synchronizing HEL1OS and SoLEXS observations")
        report_df = synchronize_datasets(hel1os_data, solexs_data)
        save_synchronization_report(report_df)

        print("[SUCCESS] Preprocessing workflow completed successfully")
        return 0

    except Exception as exc:
        print(f"[ERROR] Preprocessing workflow failed: {exc}", file=sys.stderr)
        logger.exception("Preprocessing workflow failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
