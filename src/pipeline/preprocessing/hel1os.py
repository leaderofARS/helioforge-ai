"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

HEL1OS Scientific Preprocessing Pipeline

This module coordinates the complete preprocessing
workflow for HEL1OS observations.
==========================================================
"""

from __future__ import annotations

from src.pipeline.preprocessing.config import HEL1OS_DIR
from src.utils.config import PATH_CFG

from src.pipeline.preprocessing.fits_reader import (
    read_event_file,
    read_gti_file,
    read_housekeeping_file,
)

from src.pipeline.preprocessing.metadata import (
    extract_directory_metadata,
)

from src.pipeline.preprocessing.validation import (
    validate_dataset,
    validate_time_column,
    validate_negative_values,
)

from src.utils.preprocessing_utils import (
    print_heading,
    success,
)


# ==========================================================
# Locate HEL1OS Files
# ==========================================================

def locate_hel1os_files():
    """
    Locate valid HEL1OS observations and apply the dataset budget.

    Directory layout (55 GB dataset on EC2):
        hel1os/
          hel1os_2026Jun26T<stamp>/          ← download batch
            HLS_<date>_<dur>sec_lev1_V{111,211}/     ← observation (top)
              <YYYY>/<MM>/<DD>/
                HLS_<date>_<dur>sec_lev1_V{111,211}/  ← repeated (actual data)
                  events/  evt.fits
                  aux/     gticdte1.fits  hk.fits

    Budget mode (data_paths.yaml  dataset.mode = 'budget'):
        Observations are sorted alphabetically (reproducible) and
        accumulated until the cumulative raw-file size reaches
        dataset.hel1os_gb (32.4 GB for the 55 GB run).
    """

    print_heading("Locating HEL1OS Files")

    budget = PATH_CFG.dataset
    byte_ceiling = budget.hel1os_bytes if budget.is_budget_mode else None

    if budget.is_budget_mode:
        print(
            f"[Budget] mode=budget  "
            f"ceiling={budget.hel1os_gb:.1f} GB ({byte_ceiling:,} bytes)"
        )

    # ── Step 1: collect all deduplicated valid observations ──────────────
    # rglob matches both the top-level and the deep YYYY/MM/DD copy of each
    # observation name.  We deduplicate by name, keeping the first candidate
    # that has all three required FITS files.

    seen: dict[str, dict] = {}   # obs_name → record

    for candidate in sorted(HEL1OS_DIR.rglob("HLS_*_lev1_V*")):

        if not candidate.is_dir():
            continue

        obs_name = candidate.name
        if obs_name in seen:
            continue

        event        = next(candidate.rglob("evt.fits"),       None)
        gti          = next(candidate.rglob("gticdte1.fits"),  None)
        housekeeping = next(candidate.rglob("hk.fits"),        None)

        if not (event and gti and housekeeping):
            continue

        e_sz = event.stat().st_size
        g_sz = gti.stat().st_size
        h_sz = housekeeping.stat().st_size

        if e_sz == 0 or g_sz == 0 or h_sz == 0:
            print(f"[Skip] empty files in: {obs_name}")
            continue

        seen[obs_name] = {
            "name":         obs_name,
            "path":         candidate,
            "event":        event,
            "gti":          gti,
            "housekeeping": housekeeping,
            "_size_bytes":  e_sz + g_sz + h_sz,   # raw FITS size for budget tracking
        }

    all_observations = list(seen.values())
    print(f"[HEL1OS] Total valid observations found: {len(all_observations)}")

    # ── Step 2: apply budget ceiling ─────────────────────────────────────
    if not budget.is_budget_mode or byte_ceiling is None:
        observations = all_observations
    else:
        observations = []
        accumulated = 0
        for obs in all_observations:          # already sorted alphabetically
            accumulated += obs["_size_bytes"]
            observations.append(obs)
            if accumulated >= byte_ceiling:
                break

        used_gb = accumulated / 1024 ** 3
        print(
            f"[Budget] Selected {len(observations)} / {len(all_observations)} "
            f"observations  ({used_gb:.2f} GB / {budget.hel1os_gb:.1f} GB ceiling)"
        )

    # ── Step 3: strip internal budget key before returning ───────────────
    for obs in observations:
        obs.pop("_size_bytes", None)

    print(f"\n[HEL1OS] Using {len(observations)} observations for preprocessing.")

    if not observations:
        raise FileNotFoundError(
            f"No valid HEL1OS observations found under {HEL1OS_DIR}.\n"
            "Expected structure: hel1os/<batch>/HLS_*_lev1_V*/<YYYY>/<MM>/<DD>/"
            "HLS_*_lev1_V*/{events/evt.fits, aux/gticdte1.fits, aux/hk.fits}"
        )

    success("HEL1OS files located.")
    return observations

# ==========================================================
# Stream & Validate HEL1OS Datasets (Memory Efficient)
# ==========================================================

import gc


from src.pipeline.preprocessing.config import HEL1OS_DIR, PROCESSED_DIR


def load_and_validate_hel1os(observations):
    """
    Stream each HEL1OS observation: load, validate, extract window bounds,
    save processed event.csv to PROCESSED_DIR, and free DataFrames.
    """
    print_heading("Loading & Validating HEL1OS Datasets")

    summary_records = []

    for i, obs in enumerate(observations, start=1):
        observation_name = obs["name"]
        print(f"\nProcessing Observation {i}/{len(observations)}: {observation_name}")

        try:
            event_df = read_event_file(obs["event"])
            gti_df = read_gti_file(obs["gti"])
            hk_df = read_housekeeping_file(obs["housekeeping"])

            # Validate
            validate_dataset(event_df)
            validate_time_column(event_df, "mjd")
            validate_negative_values(event_df, "ener")

            validate_dataset(hk_df)
            validate_time_column(hk_df, "mjd")

            # Save processed event.csv for Stage 3 (Features/Ingest)
            out_dir = PROCESSED_DIR / "hel1os" / observation_name
            out_dir.mkdir(parents=True, exist_ok=True)
            event_df.to_csv(out_dir / "event.csv", index=False)

            # Extract window bounds and stats
            summary_records.append({
                "name": observation_name,
                "path": obs["path"],
                "event_start_mjd": event_df["mjd"].min() if not event_df.empty else 0.0,
                "event_end_mjd": event_df["mjd"].max() if not event_df.empty else 0.0,
                "hk_start_mjd": hk_df["mjd"].min() if not hk_df.empty else 0.0,
                "hk_end_mjd": hk_df["mjd"].max() if not hk_df.empty else 0.0,
                "num_event": len(event_df),
                "num_gti": len(gti_df),
                "num_hk": len(hk_df),
            })

            del event_df, gti_df, hk_df
            gc.collect()

        except Exception as exc:
            print(f"\n[WARNING] Skipping corrupt/truncated observation '{observation_name}': {exc}")
            gc.collect()

    success("HEL1OS datasets loaded and validated successfully.")
    return summary_records


# ==========================================================
# Extract HEL1OS Metadata
# ==========================================================

def extract_hel1os_metadata():
    """
    Extract metadata from all HEL1OS scientific files.
    """
    print_heading("Extracting HEL1OS Metadata")

    patterns = ["*.fits"]
    metadata_df = extract_directory_metadata(HEL1OS_DIR, patterns)

    print(f"\nMetadata records extracted: {len(metadata_df)}")
    success("HEL1OS metadata extracted successfully.")
    return metadata_df


# ==========================================================
# HEL1OS Scientific Summary
# ==========================================================

def hel1os_summary(summary_records, metadata_df):

    print_heading("HEL1OS Scientific Summary")

    print(f"Observations Processed : {len(summary_records)}")
    print(f"Metadata Records       : {len(metadata_df)}")

    total_events = sum(d["num_event"] for d in summary_records)
    total_gti = sum(d["num_gti"] for d in summary_records)
    total_hk = sum(d["num_hk"] for d in summary_records)

    print(f"Total Event Records        : {total_events:,}")
    print(f"Total GTI Records          : {total_gti:,}")
    print(f"Total Housekeeping Records : {total_hk:,}")

    success("HEL1OS preprocessing summary generated.")


# ==========================================================
# HEL1OS Pipeline
# ==========================================================

def process_hel1os():
    """
    Execute the complete HEL1OS preprocessing pipeline.
    """
    print_heading("HEL1OS Preprocessing Pipeline")
    
    observations = locate_hel1os_files()

    datasets_summary = load_and_validate_hel1os(observations)

    metadata_df = extract_hel1os_metadata()

    hel1os_summary(datasets_summary, metadata_df)

    success("HEL1OS preprocessing completed successfully.")

    return {
        "observations": observations,
        "datasets": datasets_summary,
        "metadata": metadata_df
    }


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    process_hel1os()