"""
scripts/features.py
────────────────────
Stage 3 — Feature Engineering (Production Mode)

Loads all processed observations, runs the rolling-window feature
extractor at 1 Hz, concatenates every observation's per-second rows
into one unified feature matrix, applies variance + correlation filters,
then exports:

  selected_features.csv / .parquet / .xlsx   (F=38 selected, T rows)
  all_features.csv      / .parquet / .xlsx   (F=79 raw,      T rows)

Shape contract required by the production TCN:
  selected_features.csv → (T_total, 38+1)   columns: TIME + 38 features
  all_features.csv      → (T_total, 79+1)   columns: TIME + 79 features

Where T_total = sum of per-second rows across all observations.
With context_seconds=60 and stride=1, each observation of length L
contributes (L - 60) rows, so T_total >> 512 (the TCN window size).

Run:
    python scripts/features.py
    python scripts/features.py --context 60 --stride 1
    python scripts/features.py --context 60 --stride 4   # faster, sparser
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.rolling_feature_extractor import RollingFeatureExtractor
from src.features.feature_selector import FeatureSelector
from src.pipeline.ingestion.observation_loader import ObservationLoader
from src.utils.config import CONFIG, PATH_CFG


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, str(CONFIG["logging"]["level"]).upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("helioforge.features")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HelioForge Stage 3 — Rolling Feature Engineering"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=60,
        metavar="SEC",
        help=(
            "Rolling context window in seconds. "
            "Features at each second t are computed from [t-half, t+half]. "
            "Default: 60"
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        metavar="SEC",
        help=(
            "Step between output rows in seconds. "
            "stride=32  → one row per 32 seconds (default, fast, ~2700 rows/obs). "
            "stride=1   → one row per second (maximum resolution, very slow). "
            "Default: 32"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for feature CSVs. Default: {PATH_CFG.features.root}",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    logger = configure_logging()

    out_dir = args.output_dir or PATH_CFG.features.root
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  HELIO-FORGE  |  STAGE 3 — FEATURE ENGINEERING")
    print("=" * 60)
    print(f"  Context window : {args.context} seconds")
    print(f"  Stride         : {args.stride} second(s) per row")
    print(f"  Output dir     : {out_dir}")
    print("=" * 60 + "\n")

    try:
        # ── Load all processed observations ──────────────────────────────────
        processed_dir = PATH_CFG.preprocessing.processed
        logger.info("Loading observations from %s", processed_dir)
        loader    = ObservationLoader(processed_dir)
        extractor = RollingFeatureExtractor(
            context_seconds=args.context,
            stride=args.stride,
        )

        all_dfs = []
        skipped = 0
        total_len = 0

        for idx, obs in enumerate(loader.load_all(), start=1):
            obs_id = f"{obs['solexs_id']}/{obs['hel1os_id']}"
            t_len = len(obs["soft_signal"])
            total_len += t_len
            print(f"  [{idx:>4}] Processing: {obs_id} (T={t_len})")

            try:
                df = extractor.extract(
                    soft_signal=obs["soft_signal"],
                    hard_signal=obs["hard_signal"],
                    timestamps=obs["timestamps"],
                    observation_id=obs_id,
                )
                if df.empty:
                    print(f"         → SKIP (observation too short or no valid windows)")
                    skipped += 1
                    continue

                all_dfs.append(df)
                print(f"         → {len(df):,} rows × {len(df.columns)} cols")

            except Exception as exc:
                logger.warning("Observation %s failed: %s", obs_id, exc)
                skipped += 1
                continue

        if not all_dfs:
            raise RuntimeError(
                "No feature rows were produced. Check that processed observations "
                "exist under:\n  " + str(processed_dir) +
                "\nAnd that each observation is longer than the context window "
                f"({args.context} seconds)."
            )

        # ── Concatenate all observations into one matrix ──────────────────────
        print(f"\n  Concatenating {len(all_dfs)} observation DataFrames …")
        full_df = pd.concat(all_dfs, ignore_index=True)

        # Drop the observation_id column before feature selection
        # (it's a string — not a feature)
        meta_cols = ["TIME", "observation_id"]
        feature_cols = [c for c in full_df.columns if c not in meta_cols]

        print(f"\n  Raw feature matrix shape : {full_df.shape}")
        print(f"  Rows (T timesteps total) : {len(full_df):,}")
        print(f"  Columns (F raw features) : {len(feature_cols)}")

        if len(full_df) < 512:
            raise RuntimeError(
                f"Total rows ({len(full_df)}) is less than the TCN window size (512). "
                "Ensure your observations are longer than the context window and "
                "that stride is small enough."
            )

        # ── Export all_features (raw, before selection) ──────────────────────
        logger.info("Exporting all_features …")
        all_feat_df = full_df.copy()
        all_feat_df.to_csv(out_dir / "all_features.csv", index=False)
        try:
            all_feat_df.to_parquet(out_dir / "all_features.parquet", index=False)
            all_feat_df.to_excel(out_dir / "all_features.xlsx", index=False)
        except Exception:
            pass
        print(f"\n  Saved all_features.*  →  shape {all_feat_df.shape}")

        # ── Feature selection: variance + correlation filter ──────────────────
        print("\n  Running feature selection (variance + correlation filter) …")
        feat_only_df = full_df[feature_cols].copy()
        selector     = FeatureSelector()
        selected_df, _, n_original, n_variance, n_selected = selector.run(
            feat_only_df,
            use_feature_importance=False,
        )

        # Re-attach TIME and observation_id metadata columns to selected features
        meta_df = full_df[["TIME", "observation_id"]].reset_index(drop=True)
        final_df = pd.concat(
            [meta_df, selected_df.reset_index(drop=True)],
            axis=1,
        )


        # ── Export selected_features ──────────────────────────────────────────
        logger.info("Exporting selected_features …")
        final_df.to_csv(out_dir / "selected_features.csv", index=False)
        try:
            final_df.to_parquet(out_dir / "selected_features.parquet", index=False)
            final_df.to_excel(out_dir / "selected_features.xlsx", index=False)
        except Exception:
            pass

        # ── Summary ──────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  FEATURE ENGINEERING COMPLETE")
        print("=" * 60)
        print(f"  Observations processed : {len(all_dfs)}")
        print(f"  Observations skipped   : {skipped}")
        print(f"  Total timestep rows T  : {len(final_df):,}")
        print(f"  Raw features F         : {n_original}")
        print(f"  After variance filter  : {n_variance}")
        print(f"  After corr filter  F'  : {n_selected}   (selected_features.*)")
        print(f"  Output dir             : {out_dir}")
        print()
        print("  Next step:")
        print("    python scripts/build_windows.py --mode features \\")
        print(f"      --features-file {out_dir / 'selected_features.csv'}")
        print("=" * 60)

        return 0

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        logger.exception("Feature engineering failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
