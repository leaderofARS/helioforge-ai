"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

SoLEXS Scientific Preprocessing Pipeline

This module coordinates the preprocessing
workflow for SoLEXS observations.
==========================================================
"""

from __future__ import annotations

from src.pipeline.preprocessing.config import SOLEXS_DIR
from src.utils.config import PATH_CFG

from src.pipeline.preprocessing.fits_reader import (
    read_lightcurve,
    read_gti_file,
    read_spectrum,
)

from src.pipeline.preprocessing.metadata import (
    extract_directory_metadata,
)

from src.pipeline.preprocessing.validation import (
    validate_dataset,
    validate_time_column,
    validate_numeric_column,
)

from src.utils.preprocessing_utils import (
    print_heading,
    success,
)


# ==========================================================
# Locate SoLEXS Files
# ==========================================================

def locate_solexs_files():
    """
    Locate valid SoLEXS observations and apply the dataset budget.

    Each observation is an SDD2/ folder containing:
        *.lc    — lightcurve
        *.gti   — good time intervals
        *.pi    — spectrum

    Budget mode (data_paths.yaml  dataset.mode = 'budget'):
        Observations are sorted alphabetically by parent folder name
        (reproducible) and accumulated until the cumulative raw-file
        size reaches dataset.solexs_gb (22.6 GB for the 55 GB run).
    """

    print_heading("Locating SoLEXS Files")

    budget = PATH_CFG.dataset
    byte_ceiling = budget.solexs_bytes if budget.is_budget_mode else None

    if budget.is_budget_mode:
        print(
            f"[Budget] mode=budget  "
            f"ceiling={budget.solexs_gb:.1f} GB ({byte_ceiling:,} bytes)"
        )

    # ── Step 1: collect all valid observations ─────────────────────────────
    # Sort by parent folder name for reproducibility.
    all_sdd2 = sorted(SOLEXS_DIR.rglob("SDD2"), key=lambda d: d.parent.name)

    all_observations = []

    for sdd2_dir in all_sdd2:

        lightcurve = next(sdd2_dir.glob("*.lc"),  None)
        gti        = next(sdd2_dir.glob("*.gti"), None)
        spectrum   = next(sdd2_dir.glob("*.pi"),  None)

        if not (lightcurve and gti and spectrum):
            print(f"[Skip] incomplete observation: {sdd2_dir.parent.name}")
            continue

        lc_sz = lightcurve.stat().st_size
        g_sz  = gti.stat().st_size
        s_sz  = spectrum.stat().st_size

        if lc_sz == 0 or g_sz == 0 or s_sz == 0:
            print(f"[Skip] empty files in: {sdd2_dir.parent.name}")
            continue

        all_observations.append({
            "lightcurve":   lightcurve,
            "gti":          gti,
            "spectrum":     spectrum,
            "_size_bytes":  lc_sz + g_sz + s_sz,
        })

    print(f"[SoLEXS] Total valid observations found: {len(all_observations)}")

    # ── Step 2: apply budget ceiling ────────────────────────────────────
    if not budget.is_budget_mode or byte_ceiling is None:
        observations = all_observations
    else:
        observations = []
        accumulated = 0
        for obs in all_observations:      # already sorted alphabetically
            accumulated += obs["_size_bytes"]
            observations.append(obs)
            if accumulated >= byte_ceiling:
                break

        used_gb = accumulated / 1024 ** 3
        print(
            f"[Budget] Selected {len(observations)} / {len(all_observations)} "
            f"observations  ({used_gb:.2f} GB / {budget.solexs_gb:.1f} GB ceiling)"
        )

    # ── Step 3: strip internal budget key before returning ───────────────
    for obs in observations:
        obs.pop("_size_bytes", None)

    print(f"\n[SoLEXS] Using {len(observations)} observations for preprocessing.")

    if not observations:
        raise FileNotFoundError(
            f"No valid SoLEXS observations found under {SOLEXS_DIR}.\n"
            "Expected structure: solexs/<observation>/SDD2/{*.lc, *.gti, *.pi}"
        )

    success("SoLEXS files located.")
    return observations


# ==========================================================
# Stream & Validate SoLEXS Datasets (Memory Efficient)
# ==========================================================

import gc


def load_and_validate_solexs(observations):
    """
    Stream each SoLEXS observation: load, validate, extract window bounds,
    and free DataFrames immediately to conserve RAM.
    """
    print_heading("Loading & Validating SoLEXS Datasets")

    summary_records = []

    for i, obs in enumerate(observations, start=1):
        observation_name = obs["lightcurve"].parent.parent.name
        print(f"\nProcessing Observation {i}/{len(observations)}: {observation_name}")

        try:
            lc_df = read_lightcurve(obs["lightcurve"])
            gti_df = read_gti_file(obs["gti"])
            spectrum_df = read_spectrum(obs["spectrum"])

            # Validate
            validate_dataset(lc_df)
            validate_time_column(lc_df, "TIME")
            validate_numeric_column(lc_df, "COUNTS")

            validate_dataset(gti_df)
            validate_time_column(gti_df, "START")

            # Extract window bounds and stats
            summary_records.append({
                "name": observation_name,
                "path": obs["lightcurve"].parent.parent,
                "lc_start_time": lc_df["TIME"].min() if not lc_df.empty else 0.0,
                "lc_end_time": lc_df["TIME"].max() if not lc_df.empty else 0.0,
                "gti_start_time": gti_df["START"].min() if not gti_df.empty else 0.0,
                "gti_end_time": gti_df["STOP"].max() if not gti_df.empty else 0.0,
                "num_lc": len(lc_df),
                "num_gti": len(gti_df),
                "num_spectrum": len(spectrum_df),
            })

            del lc_df, gti_df, spectrum_df
            gc.collect()

        except Exception as exc:
            print(f"\n[WARNING] Skipping corrupt/truncated observation '{observation_name}': {exc}")
            gc.collect()

    success("SoLEXS datasets loaded and validated successfully.")
    return summary_records


# ==========================================================
# Extract SoLEXS Metadata
# ==========================================================

def extract_solexs_metadata():
    """
    Extract metadata from all SoLEXS scientific files.
    """
    print_heading("Extracting SoLEXS Metadata")

    patterns = ["*.gti", "*.lc", "*.pi"]
    metadata_df = extract_directory_metadata(SOLEXS_DIR, patterns)

    print(f"\nMetadata records extracted: {len(metadata_df)}")
    success("SoLEXS metadata extracted successfully.")
    return metadata_df


# ==========================================================
# SoLEXS Scientific Summary
# ==========================================================

def solexs_summary(summary_records, metadata_df):
    """
    Display preprocessing summary.
    """
    print_heading("SoLEXS Scientific Summary")

    print(f"Observations Processed : {len(summary_records)}")
    print(f"Metadata Records       : {len(metadata_df)}")

    total_lc = sum(d["num_lc"] for d in summary_records)
    total_gti = sum(d["num_gti"] for d in summary_records)
    total_spec = sum(d["num_spectrum"] for d in summary_records)

    print(f"Total Light Curve Records : {total_lc:,}")
    print(f"Total GTI Records         : {total_gti:,}")
    print(f"Total Spectrum Records    : {total_spec:,}")

    success("SoLEXS preprocessing summary generated.")


# ==========================================================
# SoLEXS Pipeline
# ==========================================================

def process_solexs():
    """
    Execute complete SoLEXS preprocessing pipeline.
    """
    print_heading("SoLEXS Preprocessing Pipeline")

    observations = locate_solexs_files()

    datasets_summary = load_and_validate_solexs(observations)

    metadata_df = extract_solexs_metadata()

    solexs_summary(datasets_summary, metadata_df)
    
    success("SoLEXS preprocessing completed successfully.")

    return {
        "observations": observations,
        "datasets": datasets_summary,
        "metadata": metadata_df
    }


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    process_solexs()

