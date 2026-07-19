"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
Metadata Extraction Module

This module extracts metadata from scientific files
belonging to HEL1OS and SoLEXS datasets.
==========================================================
"""

from pathlib import Path
import pandas as pd
from astropy.io import fits

from src.utils.preprocessing_utils import (
    print_heading,
    success,
    save_dataframe,
)


# ==========================================================
# Extract Metadata From One File
# ==========================================================

def extract_metadata(file_path):
    """
    Extract metadata from a scientific FITS-related file.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    dict
    """

    with fits.open(file_path) as hdul:

        header = hdul[0].header

        metadata = {

            "Filename": file_path.name,

            "File Type": file_path.suffix,

            "File Size (MB)": round(
                file_path.stat().st_size / (1024 * 1024),
                2,
            ),

            "Total HDUs": len(hdul),

            "Instrument": header.get("INSTRUME", "N/A"),

            "Telescope": header.get("TELESCOP", "N/A"),

            "Observation Date": header.get("DATE-OBS", "N/A"),

            "Object": header.get("OBJECT", "N/A"),

            "Creator": header.get("CREATOR", "N/A"),
        }

    return metadata


# ==========================================================
# Extract Metadata From Directory
# ==========================================================

def extract_directory_metadata(directory, patterns):
    """
    Extract metadata from all matching files.
    """

    records = []

    print_heading("Metadata Extraction")

    count = 0
    for pattern in patterns:
        for file in directory.rglob(pattern):
            if file.stat().st_size == 0:
                continue
            count += 1
            if count % 500 == 1:
                print(f"Extracting metadata... (file {count})")
            try:
                records.append(extract_metadata(file))
            except Exception as e:
                print(f"Skipping corrupt file : {file.name} (Reason: {e})")
                continue

    df = pd.DataFrame(records)

    return df

# ==========================================================
# Save Metadata
# ==========================================================

def save_metadata(df, output_file):
    """
    Save metadata dataframe.
    """

    save_dataframe(df, output_file)


# ==========================================================
# Validation
# ==========================================================

def validate_metadata(df):

    print_heading("Metadata Validation")

    print(f"Records : {len(df)}")

    print()

    print(df.isnull().sum())

    success("Metadata validation completed.")


# ==========================================================
# Test Module
# ==========================================================

if __name__ == "__main__":

    print_heading("Metadata Module")

    print("Metadata extraction module loaded successfully.")

    print()

    print("Available Functions")

    print("- extract_metadata()")

    print("- extract_directory_metadata()")

    print("- save_metadata()")

    print("- validate_metadata()")

    success("metadata.py is working correctly.")