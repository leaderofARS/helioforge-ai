"""
scripts/copy_outputs_to_repo.py
────────────────────────────────
Copies generated processed datasets, features, metadata CSVs, and reports
from /opt/helioforge-ai/ into the local repository directory (~/GitHub/helioforge-ai/).

Usage
-----
    python scripts/copy_outputs_to_repo.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import PATH_CFG


def copy_directory(src: Path, dst: Path, description: str) -> None:
    if not src.exists():
        print(f"[SKIP] Source directory does not exist: {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    print(f"[COPYING] {description}")
    print(f"  From : {src}")
    print(f"  To   : {dst}")

    # Copy files and subdirectories
    for item in src.glob("**/*"):
        if item.is_file():
            rel_path = item.relative_to(src)
            target_path = dst / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_path)

    print(f"[COMPLETED] {description} copied successfully.\n")


def main() -> int:
    print("=" * 60)
    print("HELIO-FORGE AI  |  COPY OUTPUTS TO REPOSITORY")
    print("=" * 60)
    print(f"Source Root : {PATH_CFG.dataset_root}")
    print(f"Target Repo : {REPO_ROOT}")
    print("=" * 60 + "\n")

    # Copy metadata CSVs
    copy_directory(
        src=PATH_CFG.metadata.root,
        dst=REPO_ROOT / "data" / "metadata",
        description="Metadata CSVs (hel1os, solexs, sync_report)",
    )

    # Copy engineered features
    copy_directory(
        src=PATH_CFG.features.root,
        dst=REPO_ROOT / "data" / "features",
        description="Engineered Features (CSV, Parquet, Excel)",
    )

    # Copy reports & figures
    copy_directory(
        src=PATH_CFG.reports.root,
        dst=REPO_ROOT / "reports",
        description="Evaluation Reports & Figures",
    )

    # Copy window tensors (.pt)
    copy_directory(
        src=PATH_CFG.windows.root,
        dst=REPO_ROOT / "data" / "windows",
        description="Sliding Window Sequence Tensors (train.pt, val.pt, test.pt)",
    )

    print("=" * 60)
    print("[SUCCESS] All outputs copied into repository at:", REPO_ROOT)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
