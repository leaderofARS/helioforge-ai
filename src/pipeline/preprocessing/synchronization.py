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
    Compare HEL1OS and SoLEXS observation windows using memory-efficient summary records.
    """
    print_heading("Observation Window Comparison")

    records = []

    for hel_obs, slx_obs in zip(
        hel1os_data["datasets"],
        solexs_data["datasets"]
    ):
        record = {
            "HEL1OS Observation": hel_obs["name"],
            "SoLEXS Observation": slx_obs["name"],
            "HEL1OS Event Start (MJD)": hel_obs["event_start_mjd"],
            "HEL1OS Event End (MJD)": hel_obs["event_end_mjd"],
            "HEL1OS HK Start (MJD)": hel_obs["hk_start_mjd"],
            "HEL1OS HK End (MJD)": hel_obs["hk_end_mjd"],
            "SoLEXS LC Start (Unix)": slx_obs["lc_start_time"],
            "SoLEXS LC End (Unix)": slx_obs["lc_end_time"],
            "SoLEXS GTI Start (Unix)": slx_obs["gti_start_time"],
            "SoLEXS GTI End (Unix)": slx_obs["gti_end_time"],
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