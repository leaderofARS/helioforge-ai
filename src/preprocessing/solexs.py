"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

SoLEXS Scientific Preprocessing Pipeline

This module coordinates the preprocessing
workflow for SoLEXS observations.
==========================================================
"""

from src.preprocessing.config import SOLEXS_DIR

from src.preprocessing.fits_reader import (
    read_lightcurve,
    read_gti_file,
    read_spectrum
)

from src.preprocessing.metadata import (
    extract_directory_metadata
)

from src.preprocessing.validation import (
    validate_dataset,
    validate_time_column,
    validate_numeric_column
)

from src.utils.preprocessing_utils import (
    print_heading,
    success
)


# ==========================================================
# Locate SoLEXS Files
# ==========================================================

def locate_solexs_files():
    """
    Locate all valid SoLEXS observations.

    Each observation should contain:

        SDD2/
            *.lc
            *.gti
            *.pi

    Empty (0-byte) files are automatically skipped.
    """

    print_heading("Locating SoLEXS Files")

    observations = []

    for sdd2_dir in SOLEXS_DIR.rglob("SDD2"):

        lightcurve = next(sdd2_dir.glob("*.lc"), None)
        gti = next(sdd2_dir.glob("*.gti"), None)
        spectrum = next(sdd2_dir.glob("*.pi"), None)

        # Check all required files exist
        if not (lightcurve and gti and spectrum):

            print(f"Skipping incomplete observation: {sdd2_dir.parent.name}")
            continue

        # Skip empty / corrupted files
        if (
            lightcurve.stat().st_size == 0
            or gti.stat().st_size == 0
            or spectrum.stat().st_size == 0
        ):

            print(f"Skipping empty observation: {sdd2_dir.parent.name}")
            continue

        observations.append({

            "lightcurve": lightcurve,
            "gti": gti,
            "spectrum": spectrum

        })

    print(f"\nFound {len(observations)} complete SoLEXS observations.")

    if not observations:

        raise FileNotFoundError(
            "No valid SoLEXS observations found."
        )

    success("SoLEXS files located.")

    return observations


# ==========================================================
# Load SoLEXS Datasets
# ==========================================================

def load_solexs(observations):
    """
    Load all SoLEXS observations.
    """

    print_heading("Loading SoLEXS Datasets")

    datasets = []

    for i, obs in enumerate(observations, start=1):

        observation_name = obs["lightcurve"].parent.parent.name
        print(f"\nLoading Observation {i}: {observation_name}")

        lc_df = read_lightcurve(obs["lightcurve"])
        gti_df = read_gti_file(obs["gti"])
        spectrum_df = read_spectrum(obs["spectrum"])

        datasets.append({
            "name": observation_name,
            "path": obs["lightcurve"].parent.parent,
            "lightcurve": lc_df,
            "gti": gti_df,
            "spectrum": spectrum_df
        })

    print("\n--------------------------------------------")
    print(f"Loaded {len(datasets)} SoLEXS observations.")
    print("--------------------------------------------")

    success("SoLEXS datasets loaded successfully.")

    return datasets
        

# ==========================================================
# Validate SoLEXS Datasets
# ==========================================================

def validate_solexs(datasets):
    """
    Validate every SoLEXS observation.
    """

    print_heading("Validating SoLEXS Datasets")

    for i, dataset in enumerate(datasets, start=1):

        print(f"\nValidating {dataset['name']}")

        lc_df = dataset["lightcurve"]
        gti_df = dataset["gti"]

        print("\nLight Curve Validation")

        validate_dataset(lc_df)
        validate_time_column(lc_df, "TIME")
        validate_numeric_column(lc_df, "COUNTS")

        print("\nGTI Validation")

        validate_dataset(gti_df)
        validate_time_column(gti_df, "START")

    success("SoLEXS validation completed.")

# ==========================================================
# Extract SoLEXS Metadata
# ==========================================================

def extract_solexs_metadata():
    """
    Extract metadata from all SoLEXS scientific files.
    """

    print_heading("Extracting SoLEXS Metadata")

    patterns = [
        "*.gti",
        "*.lc",
        "*.pi"
    ]

    metadata_df = extract_directory_metadata(
        SOLEXS_DIR,
        patterns
    )

    print()
    print(metadata_df)

    success("SoLEXS metadata extracted successfully.")

    return metadata_df


# ==========================================================
# SoLEXS Scientific Summary
# ==========================================================

def solexs_summary(datasets, metadata_df):
    """
    Display preprocessing summary.
    """

    print_heading("SoLEXS Scientific Summary")

    print(f"Observations Processed : {len(datasets)}")
    print(f"Metadata Records       : {len(metadata_df)}")

    total_lc = sum(len(d["lightcurve"]) for d in datasets)
    total_gti = sum(len(d["gti"]) for d in datasets)
    total_spec = sum(len(d["spectrum"]) for d in datasets)

    print(f"Total Light Curve Records : {total_lc}")
    print(f"Total GTI Records         : {total_gti}")
    print(f"Total Spectrum Records    : {total_spec}")

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

    datasets = load_solexs(observations)

    validate_solexs(datasets)

    metadata_df = extract_solexs_metadata()

    solexs_summary(datasets, metadata_df)
    
    success("SoLEXS preprocessing completed successfully.")

    return {

        "observations": observations,

        "datasets": datasets,

        "metadata": metadata_df

    }


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    process_solexs()

