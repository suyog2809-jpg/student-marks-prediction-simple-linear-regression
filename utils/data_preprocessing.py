"""Data Preprocessing and Validation Utility Module for Student Marks Prediction System.

This module provides robust, modular data engineering functions for:
- Dataset loading and CSV parsing
- Schema verification and column validation
- Missing value, duplicate, and anomalous data detection
- Range validation and type coercion
- Descriptive statistics and correlation analysis
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd

# Core Project Constants - Enforces strict Single Variable Regression
REQUIRED_COLUMNS = ["Student_ID", "Study_Hours", "Final_Marks"]
FEATURE_COLUMN = "Study_Hours"
TARGET_COLUMN = "Final_Marks"

HOURS_MIN = 0.0
HOURS_MAX = 24.0
MARKS_MIN = 0.0
MARKS_MAX = 100.0

# Supported interactive application input bounds (consistent across UI, inputs, and docs)
SUPPORTED_HOURS_MIN = 0.0
SUPPORTED_HOURS_MAX = 12.0


def validate_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    """Verify that the dataframe contains all required columns.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Tuple of (is_valid: bool, missing_columns: list).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return (len(missing) == 0, missing)


def load_and_clean(csv_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """Load dataset from disk, validate schema, remove anomalies, and clean records.

    Preprocessing Steps:
    1. Verify file exists and read CSV.
    2. Confirm required schema (Student_ID, Study_Hours, Final_Marks).
    3. Type coercion to numeric for Study_Hours and Final_Marks.
    4. Remove null or NaN values.
    5. Deduplicate records.
    6. Filter out-of-bound records (Study_Hours within [0, 24], Final_Marks within [0, 100]).

    Args:
        csv_path: Path to student_marks_dataset.csv.

    Returns:
        Tuple of (cleaned_df: pd.DataFrame, audit_report: dict).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    df_raw = pd.read_csv(path)
    rows_initial = len(df_raw)

    is_valid, missing_cols = validate_schema(df_raw)
    if not is_valid:
        raise ValueError(
            f"Schema validation failed. Missing required column(s): {missing_cols}"
        )

    # Work on a copy of required columns only
    df = df_raw[REQUIRED_COLUMNS].copy()

    # Numeric coercion
    df[FEATURE_COLUMN] = pd.to_numeric(df[FEATURE_COLUMN], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    # Detect nulls
    nulls_detected = int(df[[FEATURE_COLUMN, TARGET_COLUMN]].isna().any(axis=1).sum())
    df = df.dropna(subset=[FEATURE_COLUMN, TARGET_COLUMN])

    # Detect duplicates
    duplicates_detected = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Range bound filtering
    valid_hours = df[FEATURE_COLUMN].between(HOURS_MIN, HOURS_MAX)
    valid_marks = df[TARGET_COLUMN].between(MARKS_MIN, MARKS_MAX)
    valid_mask = valid_hours & valid_marks
    out_of_bounds_detected = int((~valid_mask).sum())
    df = df[valid_mask].reset_index(drop=True)

    rows_final = len(df)
    rows_removed = rows_initial - rows_final

    audit_report = {
        "rows_initial": rows_initial,
        "rows_final": rows_final,
        "rows_removed": rows_removed,
        "nulls_detected": nulls_detected,
        "duplicates_detected": duplicates_detected,
        "out_of_bounds_detected": out_of_bounds_detected,
        "retention_rate_pct": round((rows_final / rows_initial) * 100, 2) if rows_initial > 0 else 0.0,
    }

    return df, audit_report


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Union[float, int, pd.DataFrame]]:
    """Compute comprehensive descriptive statistics and correlation.

    Args:
        df: Cleaned student marks DataFrame.

    Returns:
        Dictionary containing summary table and statistical indicators.
    """
    stats_df = df[[FEATURE_COLUMN, TARGET_COLUMN]].describe().T
    stats_df["skewness"] = [
        float(df[FEATURE_COLUMN].skew()),
        float(df[TARGET_COLUMN].skew()),
    ]

    pearson_corr = float(df[FEATURE_COLUMN].corr(df[TARGET_COLUMN]))

    return {
        "record_count": int(len(df)),
        "hours_mean": float(df[FEATURE_COLUMN].mean()),
        "hours_std": float(df[FEATURE_COLUMN].std()),
        "hours_min": float(df[FEATURE_COLUMN].min()),
        "hours_max": float(df[FEATURE_COLUMN].max()),
        "marks_mean": float(df[TARGET_COLUMN].mean()),
        "marks_std": float(df[TARGET_COLUMN].std()),
        "marks_min": float(df[TARGET_COLUMN].min()),
        "marks_max": float(df[TARGET_COLUMN].max()),
        "pearson_correlation": pearson_corr,
        "stats_table": stats_df,
    }


def check_data_quality(df: pd.DataFrame) -> Dict[str, Union[bool, int, str]]:
    """Perform data hygiene and sanity verification.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Diagnostic verification dictionary.
    """
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    invalid_hours = int((~df[FEATURE_COLUMN].between(HOURS_MIN, HOURS_MAX)).sum())
    invalid_marks = int((~df[TARGET_COLUMN].between(MARKS_MIN, MARKS_MAX)).sum())

    is_clean = (missing_cells == 0) and (duplicate_rows == 0) and (invalid_hours == 0) and (invalid_marks == 0)

    return {
        "is_clean": is_clean,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "invalid_hours": invalid_hours,
        "invalid_marks": invalid_marks,
        "quality_status": "EXCELLENT" if is_clean else "REQUIRES_ATTENTION",
    }
