"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

HEL1OS Scientific Preprocessing Pipeline

This module coordinates the complete preprocessing
workflow for HEL1OS observations.
==========================================================
"""

from src.pipeline.preprocessing.config import HEL1OS_DIR

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
    Locate all valid HEL1OS observations.

    Each observation should contain:

        events/
            evt.fits

        aux/
            gticdte1.fits
            hk.fits

    Empty (0-byte) files are automatically skipped.
    """

    print_heading("Locating HEL1OS Files")

    # HEL1OS_DIR layout (55 GB dataset):
    #   hel1os/
    #     hel1os_2026Jun26T*/           ← download batch folder
    #       HLS_<date>_<dur>sec_lev1_V{111,211}/      ← top-level obs name
    #         <YYYY>/<MM>/<DD>/
    #           HLS_<date>_<dur>sec_lev1_V{111,211}/  ← repeated folder (actual data)
    #             events/  evt.fits
    #             aux/     gticdte1.fits  hk.fits
    #
    # rglob("HLS_*_lev1_V*") matches BOTH levels.
    # We deduplicate by observation name and keep only the dir that
    # actually contains evt.fits underneath it.

    seen: dict[str, dict] = {}   # obs_name → record

    for candidate in sorted(HEL1OS_DIR.rglob("HLS_*_lev1_V*")):

        if not candidate.is_dir():
            continue

        obs_name = candidate.name

        # Skip if we already resolved this observation to a valid record
        if obs_name in seen:
            continue

        event        = next(candidate.rglob("evt.fits"),        None)
        gti          = next(candidate.rglob("gticdte1.fits"),   None)
        housekeeping = next(candidate.rglob("hk.fits"),         None)

        # Skip if any required file is missing
        if not (event and gti and housekeeping):
            continue

        # Skip empty files
        if (
            event.stat().st_size == 0
            or gti.stat().st_size == 0
            or housekeeping.stat().st_size == 0
        ):
            print(f"Skipping empty observation: {obs_name}")
            continue

        seen[obs_name] = {
            "name":         obs_name,
            "path":         candidate,
            "event":        event,
            "gti":          gti,
            "housekeeping": housekeeping,
        }

    observations = list(seen.values())

    print(f"\nFound {len(observations)} complete HEL1OS observations.")

    if not observations:
        raise FileNotFoundError(
            f"No valid HEL1OS observations found under {HEL1OS_DIR}.\n"
            "Expected structure: hel1os/<batch>/HLS_*_lev1_V*/<YYYY>/<MM>/<DD>/"
            "HLS_*_lev1_V*/{events/evt.fits, aux/gticdte1.fits, aux/hk.fits}"
        )

    success("HEL1OS files located.")


    return observations

# ==========================================================
# Load HEL1OS Datasets
# ==========================================================

def load_hel1os(observations):

    print_heading("Loading HEL1OS Datasets")

    datasets = []

    for i, obs in enumerate(observations, start=1):

        observation_name = obs["name"]

        print(f"\nLoading Observation {i}: {observation_name}")

        event_df = read_event_file(obs["event"])
        gti_df = read_gti_file(obs["gti"])
        hk_df = read_housekeeping_file(obs["housekeeping"])

        datasets.append({

            "name": observation_name,

            "path": obs["path"],

            "event": event_df,

            "gti": gti_df,

            "housekeeping": hk_df

        })

    success("HEL1OS datasets loaded successfully.")

    return datasets

# ==========================================================
# Validate HEL1OS Datasets
# ==========================================================

def validate_hel1os(datasets):

    print_heading("Validating HEL1OS Datasets")

    for dataset in datasets:

        print(f"\nValidating {dataset['name']}")

        event_df = dataset["event"]
        hk_df = dataset["housekeeping"]

        print("\nEvent Dataset Validation")

        validate_dataset(event_df)
        validate_time_column(event_df, "mjd")
        validate_negative_values(event_df, "ener")

        print("\nHousekeeping Dataset Validation")

        validate_dataset(hk_df)
        validate_time_column(hk_df, "mjd")

    success("HEL1OS validation completed.")
    
# ==========================================================
# Extract HEL1OS Metadata
# ==========================================================

def extract_hel1os_metadata():
    """
    Extract metadata from all HEL1OS scientific files.
    """

    print_heading("Extracting HEL1OS Metadata")

    patterns = [
        "*.fits"
    ]

    metadata_df = extract_directory_metadata(
        HEL1OS_DIR,
        patterns
    )

    print()
    print(metadata_df)

    success("HEL1OS metadata extracted successfully.")

    return metadata_df


# ==========================================================
# HEL1OS Scientific Summary
# ==========================================================

def hel1os_summary(datasets, metadata_df):

    print_heading("HEL1OS Scientific Summary")

    print(f"Observations Processed : {len(datasets)}")
    print(f"Metadata Records       : {len(metadata_df)}")

    total_events = sum(len(d["event"]) for d in datasets)
    total_gti = sum(len(d["gti"]) for d in datasets)
    total_hk = sum(len(d["housekeeping"]) for d in datasets)

    print(f"Total Event Records        : {total_events}")
    print(f"Total GTI Records          : {total_gti}")
    print(f"Total Housekeeping Records : {total_hk}")

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

    datasets = load_hel1os(observations)

    validate_hel1os(datasets)

    metadata_df = extract_hel1os_metadata()

    hel1os_summary(datasets, metadata_df)

    success("HEL1OS preprocessing completed successfully.")

    return {
        "observations": observations,

        "datasets": datasets,

        "metadata": metadata_df
    }


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    process_hel1os()