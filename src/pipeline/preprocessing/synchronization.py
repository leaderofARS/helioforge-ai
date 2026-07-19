"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)

Synchronization Module

This module compares the observation windows of
HEL1OS and SoLEXS datasets and generates a
synchronization report.
==========================================================
"""

from pathlib import Path

import pandas as pd

from src.utils.config import PATH_CFG
from src.utils.preprocessing_utils import (
    print_heading,
    success,
    save_dataframe,
)


# ==========================================================
# Compare Observation Windows
# ==========================================================

def compare_observation_windows(hel1os_data, solexs_data):
    """
    Compare HEL1OS and SoLEXS observation windows.
    """

    print_heading("Observation Window Comparison")

    records = []

    for hel_obs, slx_obs in zip(
        hel1os_data["datasets"],
        solexs_data["datasets"]
    ):

        event_df = hel_obs["event"]
        hk_df = hel_obs["housekeeping"]

        lc_df = slx_obs["lightcurve"]
        gti_df = slx_obs["gti"]

        record = {

            "HEL1OS Observation": hel_obs["name"],

            "SoLEXS Observation": slx_obs["name"],

            "HEL1OS Event Start (MJD)": event_df["mjd"].min(),
            "HEL1OS Event End (MJD)": event_df["mjd"].max(),

            "HEL1OS HK Start (MJD)": hk_df["mjd"].min(),
            "HEL1OS HK End (MJD)": hk_df["mjd"].max(),

            "SoLEXS LC Start (Unix)": lc_df["TIME"].min(),
            "SoLEXS LC End (Unix)": lc_df["TIME"].max(),

            "SoLEXS GTI Start (Unix)": gti_df["START"].min(),
            "SoLEXS GTI End (Unix)": gti_df["STOP"].max()

        }

        records.append(record)

    success("Observation windows compared.")

    return pd.DataFrame(records)


# ==========================================================
# Display Synchronization Report
# ==========================================================

def synchronization_report(report_df):
    """
    Display synchronization report.
    """

    print_heading("Synchronization Report")

    print(report_df)

    success("Synchronization report generated.")

    return report_df


# ==========================================================
# Save Report
# ==========================================================

def save_synchronization_report(report_df, output_path=None):
    """
    Save synchronization report.
    """

    output_path = (
        Path(output_path)
        if output_path is not None
        else PATH_CFG.metadata.sync_report
    )
    save_dataframe(report_df, output_path)

    success("Synchronization report saved.")


# ==========================================================
# Main Synchronization Function
# ==========================================================

def synchronize_datasets(hel1os_data, solexs_data):
    """
    Execute synchronization workflow.
    """

    report_df = compare_observation_windows(
        hel1os_data,
        solexs_data
    )

    synchronization_report(report_df)

    return report_df


# ==========================================================
# Test Module
# ==========================================================

if __name__ == "__main__":

    print_heading("Synchronization Module")

    print("This module is intended to be called from main.py")

    success("synchronization.py loaded successfully.")