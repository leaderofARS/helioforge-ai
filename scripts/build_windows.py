"""
scripts/build_windows.py
────────────────────────
CLI entry point for the HelioForge HPINA sliding window generator.

Supports two modes:

  --mode raw        (default)
      Slices the raw 2-channel SoLEXS / HEL1OS flux streams into tensors
      of shape (N_windows, 2, window_size).
      Uses WindowGenerator.

  --mode features
      Implements the Chapter 1 mathematical formulation W_t ∈ R^(F × L).
      Slices a pre-built engineered feature matrix CSV into tensors of shape
      (N_windows, F, window_size) where F = 38 or 79 depending on the CSV.
      Uses MultivariateFeatureWindowGenerator.
      Requires --features-file pointing to selected_features.csv (F=38)
      or all_features.csv (F=79).

Usage examples
--------------
  # Raw 2-channel windows at default scale (512, stride 32)
  python scripts/build_windows.py

  # Raw multi-scale (256, 512, 1024)
  python scripts/build_windows.py --all-scales

  # Feature windows  F=38  default scale
  python scripts/build_windows.py --mode features \\
      --features-file /opt/helioforge-ai/data/features/selected_features.csv

  # Feature windows  F=79  all scales  custom output dir
  python scripts/build_windows.py --mode features --all-scales \\
      --features-file /opt/helioforge-ai/data/features/all_features.csv \\
      --output-dir /opt/helioforge-ai/data/windows/f79
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure repository root is on sys.path ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.HPINA.data.datasets import (
    MultivariateFeatureWindowGenerator,
    WindowGenerator,
)
from src.utils.config import PATH_CFG


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_windows.py",
        description="HelioForge HPINA — Sliding Window Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["raw", "features"],
        default="raw",
        help=(
            "Window generation mode.\n"
            "  raw      — 2-channel SoLEXS/HEL1OS flux windows  (N, 2, L)\n"
            "  features — F-channel feature matrix windows       (N, F, L)\n"
            "Default: raw"
        ),
    )

    # ── Scale ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        metavar="L",
        help="Sequence window length in timesteps (default: 512).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        metavar="S",
        help="Sliding window step stride in timesteps (default: 32).",
    )
    parser.add_argument(
        "--all-scales",
        action="store_true",
        help="Generate tensors for all scales: w256 (s=16), w512 (s=32), w1024 (s=64).",
    )

    # ── Feature mode options ───────────────────────────────────────────────────
    parser.add_argument(
        "--features-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "[features mode] Path to the feature matrix CSV.\n"
            "Examples:\n"
            "  selected_features.csv  → F=38 selected physics features\n"
            "  all_features.csv       → F=79 full candidate feature space\n"
            f"Default: {PATH_CFG.features.csv}"
        ),
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="TIME",
        metavar="COL",
        help="[features mode] Name of the time column in the feature CSV (default: TIME).",
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory to write .pt tensor files and scaler JSON.\n"
            f"Default: {PATH_CFG.windows.root}"
        ),
    )

    # ── Verbosity ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("helioforge.build_windows")

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  HELIO-FORGE AI  |  HPINA WINDOW GENERATOR")
    print("=" * 60)
    print(f"  Mode         : {args.mode.upper()}")
    if args.all_scales:
        print("  Scales       : w256 (s=16)  w512 (s=32)  w1024 (s=64)")
    else:
        print(f"  Window Size  : {args.window_size}")
        print(f"  Stride       : {args.stride}")

    output_dir = args.output_dir or PATH_CFG.windows.root
    print(f"  Output Dir   : {output_dir}")

    # ── RAW mode ──────────────────────────────────────────────────────────────
    if args.mode == "raw":
        print("  Channels     : 2  (SoLEXS COUNTS + HEL1OS energy)")
        print("=" * 60 + "\n")

        generator = WindowGenerator(
            window_size=args.window_size,
            stride=args.stride,
            output_dir=output_dir,
        )
        if args.all_scales:
            generator.generate_all_scales()
        else:
            generator.generate_all()

    # ── FEATURES mode ─────────────────────────────────────────────────────────
    else:
        features_csv = args.features_file or PATH_CFG.features.csv
        if not features_csv.exists():
            logger.error(
                "Feature CSV not found: %s\n"
                "Run  python scripts/features.py  first to generate it.",
                features_csv,
            )
            return 1

        print(f"  Features CSV : {features_csv}")
        print("=" * 60 + "\n")

        generator = MultivariateFeatureWindowGenerator(
            features_csv=features_csv,
            window_size=args.window_size,
            stride=args.stride,
            output_dir=output_dir,
            time_col=args.time_col,
        )
        if args.all_scales:
            generator.generate_all_scales()
        else:
            generator.generate_all()

    logger.info("Window generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
