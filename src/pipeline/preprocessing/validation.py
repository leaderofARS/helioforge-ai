"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
Validation Module

This module provides reusable validation functions
for HEL1OS and SoLEXS scientific datasets.
==========================================================
"""

import pandas as pd

from src.utils.preprocessing_utils import (
    print_heading,
    success,
    warning
)

# ==========================================================
# Missing Values
# ==========================================================

def validate_missing_values(df):
    """
    Check missing values in a DataFrame.
    """

    print_heading("Missing Value Validation")

    missing = df.isnull().sum()

    print(missing)

    total_missing = missing.sum()

    if total_missing == 0:
        success("No missing values found.")
    else:
        warning(f"{total_missing} missing values detected.")

    return total_missing


# ==========================================================
# Empty Dataset
# ==========================================================

def validate_empty_dataset(df):
    """
    Check whether a dataset is empty.
    """

    print_heading("Dataset Validation")

    if df.empty:
        warning("Dataset is empty.")
        return False

    print(f"Records : {len(df)}")

    success("Dataset contains valid records.")

    return True


# ==========================================================
# Time Column
# ==========================================================

def validate_time_column(df, column_name):
    """
    Validate a timestamp column.
    """

    print_heading("Timestamp Validation")

    if column_name not in df.columns:
        warning(f"{column_name} column not found.")
        return

    print("Start :", df[column_name].min())
    print("End   :", df[column_name].max())

    success("Timestamp validation completed.")


# ==========================================================
# Numeric Column
# ==========================================================

def validate_numeric_column(df, column_name):
    """
    Display statistics of a numeric column.
    """

    print_heading(f"{column_name} Statistics")

    if column_name not in df.columns:
        warning(f"{column_name} column not found.")
        return

    print(df[column_name].describe())

    success("Statistics generated.")


# ==========================================================
# Negative Values
# ==========================================================

def validate_negative_values(df, column_name):
    """
    Count negative values.
    """

    print_heading(f"{column_name} Validation")

    if column_name not in df.columns:
        warning(f"{column_name} column not found.")
        return

    negatives = (df[column_name] < 0).sum()

    print(f"Negative Values : {negatives}")

    if negatives == 0:
        success("No negative values detected.")
    else:
        warning(f"{negatives} negative values found.")

    return negatives


# ==========================================================
# Dataset Summary
# ==========================================================

def validation_summary(df):
    """
    Print a validation summary.
    """

    print_heading("Validation Summary")

    print(f"Rows            : {df.shape[0]}")
    print(f"Columns         : {df.shape[1]}")
    print(f"Missing Values  : {df.isnull().sum().sum()}")

    success("Validation completed successfully.")


# ==========================================================
# Complete Validation
# ==========================================================

def validate_dataset(df):
    """
    Run standard validation checks.
    """

    validate_empty_dataset(df)
    validate_missing_values(df)
    validation_summary(df)


# ==========================================================
# Test Module
# ==========================================================

if __name__ == "__main__":

    print_heading("Validation Module")

    print("Validation module loaded successfully.")

    print("\nAvailable Functions")

    print("- validate_missing_values()")
    print("- validate_empty_dataset()")
    print("- validate_time_column()")
    print("- validate_numeric_column()")
    print("- validate_negative_values()")
    print("- validation_summary()")
    print("- validate_dataset()")

    success("validation.py is working correctly.")