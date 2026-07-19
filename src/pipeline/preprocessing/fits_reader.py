"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
FITS Reader Module

This module contains reusable functions for reading
scientific FITS files from HEL1OS and SoLEXS datasets.

Supported Files
---------------
- Event Files (.fits)
- GTI Files (.fits / .gti)
- Housekeeping Files (.fits)
- Light Curve Files (.fits / .lc)
- Spectrum Files (.fits / .pi)
==========================================================
"""

from astropy.io import fits
import pandas as pd
import numpy as np

from src.utils.preprocessing_utils import print_heading, success


# ==========================================================
# Convert FITS Binary Table to DataFrame
# ==========================================================

def _table_to_dataframe(hdu):
    """
    Internal helper function to convert a FITS Binary Table
    into a Pandas DataFrame.
    """

    data = np.array(hdu.data)

    data = data.byteswap().view(
        data.dtype.newbyteorder("=")
    )

    return pd.DataFrame(data)


# ==========================================================
# Inspect FITS File
# ==========================================================

def inspect_fits(file_path):
    """
    Display FITS file structure.
    """

    print_heading("FITS Structure")

    with fits.open(file_path) as hdul:
        hdul.info()


# ==========================================================
# Read Header
# ==========================================================

def read_header(file_path, hdu_index=0):
    """
    Read FITS header.
    """

    with fits.open(file_path) as hdul:
        return hdul[hdu_index].header


# ==========================================================
# Read Event File
# ==========================================================

def read_event_file(file_path):
    """
    Read HEL1OS Event File.
    """
    with fits.open(file_path, ignore_missing_end=True) as hdul:
        df = _table_to_dataframe(hdul[1])
        if "recnum" in df.columns:
            df["recnum"] = df["recnum"].astype("int32")
    success("Event dataset loaded.")
    return df


# ==========================================================
# Read GTI File
# ==========================================================

def read_gti_file(file_path):
    """
    Read Good Time Interval (GTI) file.
    """
    with fits.open(file_path, ignore_missing_end=True) as hdul:
        df = _table_to_dataframe(hdul[1])
    success("GTI dataset loaded.")
    return df


# ==========================================================
# Read Housekeeping File
# ==========================================================

def read_housekeeping_file(file_path):
    """
    Read Housekeeping dataset.
    """
    with fits.open(file_path, ignore_missing_end=True) as hdul:
        df = _table_to_dataframe(hdul[1])
    success("Housekeeping dataset loaded.")
    return df


# ==========================================================
# Read Light Curve
# ==========================================================

def read_lightcurve(file_path):
    """
    Read SoLEXS Light Curve.
    """
    with fits.open(file_path, ignore_missing_end=True) as hdul:
        df = _table_to_dataframe(hdul[1])
    success("Light Curve dataset loaded.")
    return df


# ==========================================================
# Read Spectrum
# ==========================================================

def read_spectrum(file_path):
    """
    Read SoLEXS Spectrum (.pi).

    Since the spectrum contains multidimensional columns
    (CHANNEL and COUNTS), we return the FITS table directly
    instead of converting it into a DataFrame.
    """
    with fits.open(file_path, ignore_missing_end=True) as hdul:
        spectrum = hdul[1].data.copy()
    success("Spectrum dataset loaded.")
    return spectrum




# ==========================================================
# Display Columns
# ==========================================================

def show_columns(file_path):
    """
    Display FITS table columns.
    """

    print_heading("FITS Columns")

    with fits.open(file_path) as hdul:

        print(hdul[1].columns)


# ==========================================================
# Test Module
# ==========================================================

if __name__ == "__main__":

    print_heading("FITS Reader Module")

    print("Reusable FITS reader functions loaded successfully.")

    print("\nAvailable Functions")

    print("- inspect_fits()")
    print("- read_header()")
    print("- read_event_file()")
    print("- read_gti_file()")
    print("- read_housekeeping_file()")
    print("- read_lightcurve()")
    print("- read_spectrum()")
    print("- show_columns()")

    success("fits_reader.py is working correctly.")