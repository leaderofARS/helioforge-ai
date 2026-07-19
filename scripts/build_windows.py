"""
scripts/build_windows.py
────────────────────────
Script to build sliding window sequence tensors for HPINA TCN models.

Usage
-----
    python scripts/build_windows.py --window-size 512 --stride 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.HPINA.data.datasets import WindowGenerator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HelioForge HPINA — Sliding Window Generator"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Sequence window length in timesteps (default: 512)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Sliding window step stride in timesteps (default: 32)",
    )

    parser.add_argument(
        "--all-scales",
        action="store_true",
        help="Generate sequence window tensors for all scales (256, 512, 1024)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  HELIO-FORGE AI  |  HPINA WINDOW GENERATOR")
    print("=" * 60)
    if args.all_scales:
        print("  Mode        : Multi-Scale Generation (w256, w512, w1024)")
    else:
        print(f"  Window Size : {args.window_size}")
        print(f"  Stride      : {args.stride}")
    print("=" * 60 + "\n")

    generator = WindowGenerator(window_size=args.window_size, stride=args.stride)
    if args.all_scales:
        generator.generate_all_scales()
    else:
        generator.generate_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
