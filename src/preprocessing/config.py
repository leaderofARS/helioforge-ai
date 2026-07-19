"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
Project Configuration

This module stores all common project paths used
throughout the preprocessing pipeline.
==========================================================
"""

from pathlib import Path

from src.utils.config import CONFIG, get_path

PROJECT_ROOT = Path(CONFIG["paths"]["project_root"])
DATA_DIR = get_path("data")
RAW_DATA = get_path("raw")
PROCESSED_DATA = get_path("processed")
METADATA_DIR = get_path("metadata")
REPORTS_DIR = get_path("reports")
DOCS_DIR = get_path("docs")
HEL1OS_DIR = get_path("hel1os")
SOLEXS_DIR = get_path("solexs")
HEL1OS_METADATA = Path(CONFIG["files"]["hel1os_metadata"])
SOLEXS_METADATA = Path(CONFIG["files"]["solexs_metadata"])

for directory in (PROCESSED_DATA, METADATA_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print("HELIO-FORGE Configuration")
    print("=" * 60)
    print(f"Project Root      : {PROJECT_ROOT}")
    print(f"Raw Data          : {RAW_DATA}")
    print(f"HEL1OS Directory  : {HEL1OS_DIR}")
    print(f"SoLEXS Directory  : {SOLEXS_DIR}")
    print(f"Processed Data    : {PROCESSED_DATA}")
    print(f"Reports Directory : {REPORTS_DIR}")
    print(f"Documents         : {DOCS_DIR}")