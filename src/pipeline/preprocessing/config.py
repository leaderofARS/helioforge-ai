"""
src/pipeline/preprocessing/config.py
──────────────────────────────────────
Preprocessing pipeline path constants.

All paths are resolved from configs/data_paths.yaml via
src.utils.config.PATH_CFG.  This module is the single
import point for paths within the preprocessing sub-package.
"""

from src.utils.config import PATH_CFG

# ─── Raw FITS directories ─────────────────────────────────────────────────────
SOLEXS_DIR   = PATH_CFG.raw.solexs          # /opt/helioforge/raw/solexs
HEL1OS_DIR   = PATH_CFG.raw.hel1os          # /opt/helioforge/raw/hel1os

# ─── Preprocessing output directories ────────────────────────────────────────
VALIDATED_DIR    = PATH_CFG.preprocessing.validated      # /opt/helioforge/preprocessing/validated
SYNCHRONIZED_DIR = PATH_CFG.preprocessing.synchronized   # /opt/helioforge/preprocessing/synchronized
PROCESSED_DIR    = PATH_CFG.preprocessing.processed      # /opt/helioforge/preprocessing/processed

# ─── Metadata ────────────────────────────────────────────────────────────────
METADATA_DIR     = PATH_CFG.metadata.root
SOLEXS_METADATA  = PATH_CFG.metadata.solexs_metadata    # /opt/helioforge/metadata/solexs_metadata.csv
HEL1OS_METADATA  = PATH_CFG.metadata.hel1os_metadata    # /opt/helioforge/metadata/hel1os_metadata.csv
SYNC_REPORT      = PATH_CFG.metadata.sync_report        # /opt/helioforge/metadata/synchronization_report.csv

# ─── Reports ─────────────────────────────────────────────────────────────────
REPORTS_DIR  = PATH_CFG.reports.root
FIGURES_DIR  = PATH_CFG.reports.figures
TABLES_DIR   = PATH_CFG.reports.tables

# ─── Logs ────────────────────────────────────────────────────────────────────
PIPELINE_LOG = PATH_CFG.logs.pipeline

# ─── Auto-create output directories on import ────────────────────────────────
for _dir in (
    VALIDATED_DIR,
    SYNCHRONIZED_DIR,
    PROCESSED_DIR,
    METADATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("HelioForge-AI  |  Preprocessing Configuration")
    print("=" * 60)
    print(f"  SoLEXS raw       : {SOLEXS_DIR}")
    print(f"  HEL1OS raw       : {HEL1OS_DIR}")
    print(f"  Validated        : {VALIDATED_DIR}")
    print(f"  Synchronized     : {SYNCHRONIZED_DIR}")
    print(f"  Processed        : {PROCESSED_DIR}")
    print(f"  Metadata         : {METADATA_DIR}")
    print(f"  Reports          : {REPORTS_DIR}")
    print(f"  Pipeline log     : {PIPELINE_LOG}")