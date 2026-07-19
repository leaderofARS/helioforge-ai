"""
scripts/clean_outputs.py
─────────────────────────
Safely purges all generated pipeline outputs (preprocessing, features,
windows, metadata, reports, outputs, logs) while leaving raw dataset
(/opt/helioforge-ai/data/raw) 100% untouched.

Usage
-----
    python scripts/clean_outputs.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import PATH_CFG


def main() -> int:
    print("=" * 60)
    print("HELIO-FORGE AI  |  PURGE GENERATED OUTPUTS")
    print("=" * 60)
    print(f"Dataset root : {PATH_CFG.dataset_root}")
    print(f"RAW (SAFE)   : {PATH_CFG.raw.root}")
    print("=" * 60)

    # All generated output directories to reset
    targets = [
        ("Preprocessing - Validated",    PATH_CFG.preprocessing.validated),
        ("Preprocessing - Synchronized", PATH_CFG.preprocessing.synchronized),
        ("Preprocessing - Processed",    PATH_CFG.preprocessing.processed),
        ("Features Root",                PATH_CFG.features.root),
        ("Windows Root",                 PATH_CFG.windows.root),
        ("Metadata Root",                PATH_CFG.metadata.root),
        ("Reports - Figures",            PATH_CFG.reports.figures),
        ("Reports - Tables",             PATH_CFG.reports.tables),
        ("Outputs - Predictions",        PATH_CFG.outputs.predictions),
        ("Outputs - Reports",            PATH_CFG.outputs.reports),
        ("Logs Root",                    PATH_CFG.logs.root),
    ]

    for label, target_path in targets:
        # Safety assertion: NEVER delete raw data directory
        if PATH_CFG.raw.root in target_path.parents or target_path == PATH_CFG.raw.root:
            print(f"[SAFETY SKIP] Refusing to touch raw path: {target_path}")
            continue

        if target_path.exists():
            print(f"[PURGING] {label} → {target_path}")
            shutil.rmtree(target_path, ignore_errors=True)

        target_path.mkdir(parents=True, exist_ok=True)
        print(f"[RESET]   {label} → recreated empty dir: {target_path}")

    print("\n" + "=" * 60)
    print("[SUCCESS] All non-raw generated outputs successfully purged.")
    print("Raw dataset preserved untouched at:", PATH_CFG.raw.root)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
