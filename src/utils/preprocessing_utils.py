"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
preprocessing_utils.py

Common utility functions used across
the feature engineering pipeline.
==========================================================
"""

from pathlib import Path
from astropy.io import fits
import pandas as pd
import numpy as np

# Path resolution is handled by src.utils.config — not needed here


# ==========================================================
# Find Files
# ==========================================================

def find_files(directory: Path, pattern: str):
    """
    Recursively find files matching a pattern.

    Parameters
    ----------
    directory : Path
        Directory to search.

    pattern : str
        File pattern (e.g. "*.fits", "*.lc").

    Returns
    -------
    list
        List of matching files.
    """
    return list(directory.rglob(pattern))


# ==========================================================
# Open FITS File
# ==========================================================

def open_fits(file_path):
    """
    Open a FITS file safely.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    HDUList
    """
    return fits.open(file_path)


# ==========================================================
# Convert FITS Table to DataFrame
# ==========================================================

def fits_to_dataframe(hdu):
    """
    Convert a FITS binary table into a Pandas DataFrame.

    Handles NumPy byte-order conversion automatically.

    Parameters
    ----------
    hdu : BinTableHDU

    Returns
    -------
    pandas.DataFrame
    """

    data = np.array(hdu.data)

    data = data.byteswap().view(
        data.dtype.newbyteorder("=")
    )

    return pd.DataFrame(data)


# ==========================================================
# Display Dataset Summary
# ==========================================================

def dataset_summary(df, dataset_name="Dataset"):
    """
    Print a quick summary of a DataFrame.
    """

    print("=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(f"Rows            : {df.shape[0]}")
    print(f"Columns         : {df.shape[1]}")
    print(f"Missing Values  : {df.isnull().sum().sum()}")

    print("\nColumn Names")
    print(df.columns.tolist())


# ==========================================================
# Save DataFrame
# ==========================================================

def save_dataframe(df, output_path):
    """
    Save DataFrame as CSV.

    Parameters
    ----------
    df : pandas.DataFrame

    output_path : Path
    """

    df.to_csv(output_path, index=False)

    print(f"\n[SAVED] Saved successfully -> {output_path}")


# ==========================================================
# Print Section Heading
# ==========================================================

def print_heading(title):
    """
    Print a formatted heading.
    """

    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)


# ==========================================================
# Print Success Message
# ==========================================================

def success(message):
    """
    Print a success message.
    """

    print(f"\n[SUCCESS] {message}")


# ==========================================================
# Print Warning Message
# ==========================================================

def warning(message):
    """
    Print a warning message.
    """

    print(f"\n[WARNING] {message}")


# ==========================================================
# Print Error Message
# ==========================================================

def error(message):
    """
    Print an error message.
    """

    print(f"\n[ERROR] {message}")


# ==========================================================
# Test Module
# ==========================================================

if __name__ == "__main__":

    print_heading("Utility Module")

    print("Utility functions loaded successfully.")

    print("\nAvailable Functions")

    print("- find_files()")
    print("- open_fits()")
    print("- fits_to_dataframe()")
    print("- dataset_summary()")
    print("- save_dataframe()")
    print("- print_heading()")
    print("- success()")
    print("- warning()")
    print("- error()")

    success("utils.py is working correctly.")