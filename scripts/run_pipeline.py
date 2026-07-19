"""
scripts/run_pipeline.py
────────────────────────
Master entry point — run the entire HelioForge pipeline in one shot.

    Stage 1+2  preprocess.py   Raw FITS → validated/synchronised/processed
    Stage 3    features.py     Processed → feature engineering + selection
    Stage 4    ingest.py       Processed → ML-ready dataset export
    Stage 5    evaluate.py     Features → visualisations + reports

Usage
-----
    # Full pipeline
    python scripts/run_pipeline.py

    # Skip a stage
    python scripts/run_pipeline.py --skip preprocess
    python scripts/run_pipeline.py --skip ingest

    # Only run specific stages
    python scripts/run_pipeline.py --only preprocess features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import PATH_CFG


##################################################
# STAGE RUNNERS
##################################################

def stage_preprocess() -> int:
    from scripts.preprocess import main
    return main()


def stage_features() -> int:
    from scripts.features import main
    return main()


def stage_ingest() -> int:
    from scripts.ingest import main
    return main()


def stage_evaluate() -> int:
    from scripts.evaluate import main
    return main()


##################################################
# PIPELINE STAGES
##################################################

STAGES: dict[str, tuple[str, callable]] = {
    "preprocess": ("Raw FITS → Preprocessing (HEL1OS + SoLEXS + Sync)", stage_preprocess),
    "features":   ("Processed → Feature Engineering + Selection",         stage_features),
    "ingest":     ("Processed → ML Dataset Export",                        stage_ingest),
    "evaluate":   ("Features → Visualisations + Reports",                  stage_evaluate),
}


##################################################
# MAIN
##################################################

def main() -> int:

    parser = argparse.ArgumentParser(
        description="HelioForge-AI  |  Full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=list(STAGES),
        default=[],
        metavar="STAGE",
        help="Stages to skip (space-separated)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(STAGES),
        default=[],
        metavar="STAGE",
        help="Run only these stages (space-separated, in order)",
    )
    args = parser.parse_args()

    # Resolve active stages
    if args.only:
        active = {k: v for k, v in STAGES.items() if k in args.only}
    else:
        active = {k: v for k, v in STAGES.items() if k not in args.skip}

    ##################################################
    # PRINT HEADER
    ##################################################

    print()
    print("=" * 60)
    print("  HELIO-FORGE AI  |  PIPELINE RUNNER")
    print("=" * 60)
    print(f"  Dataset root : {PATH_CFG.dataset_root}")
    print(f"  Raw SoLEXS   : {PATH_CFG.raw.solexs}")
    print(f"  Raw HEL1OS   : {PATH_CFG.raw.hel1os}")
    print(f"  Processed    : {PATH_CFG.preprocessing.processed}")
    print(f"  Features     : {PATH_CFG.features.root}")
    print(f"  Reports      : {PATH_CFG.reports.root}")
    print("=" * 60)
    print()
    print("  Stages to run:")
    for name, (description, _) in active.items():
        print(f"    [{name}]  {description}")
    print()

    ##################################################
    # EXECUTE STAGES
    ##################################################

    for stage_name, (description, runner) in active.items():

        print()
        print("─" * 60)
        print(f"  STAGE: {stage_name.upper()}")
        print(f"  {description}")
        print("─" * 60)

        return_code = runner()

        if return_code != 0:
            print(
                f"\n[PIPELINE ABORTED] Stage '{stage_name}' failed "
                f"(exit code {return_code}). Stopping pipeline.",
                file=sys.stderr,
            )
            return return_code

    ##################################################
    # DONE
    ##################################################

    print()
    print("=" * 60)
    print("  HELIO-FORGE PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Features saved : {PATH_CFG.features.root}")
    print(f"  Reports saved  : {PATH_CFG.reports.root}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
