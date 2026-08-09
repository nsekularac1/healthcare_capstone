"""Preprocessing utilities for the Diabetes 130-US Hospitals dataset.

This module converts the exploratory findings from 01_diabetes_eda.ipynb
into a reusable preprocessing workflow for 30-day readmission modeling.

Key design decisions:
- Treat "?" as missing data.
- Create a binary target where <30 readmission = 1.
- Drop features with extremely high missingness.
- Preserve patient_nbr separately for leakage-aware train/test splitting.
- Remove identifiers from the predictive feature matrix.
- Group raw ICD diagnosis codes into broader clinical categories.
- Fill missing categorical values with "Unknown".
- Keep numerical preparation separate from model fitting so that sklearn
  pipelines can handle scaling/encoding without leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMN = "readmitted_30"
ORIGINAL_TARGET_COLUMN = "readmitted"

IDENTIFIER_COLUMNS = [
    "encounter_id",
    "patient_nbr",
]

# Based on the EDA results:
# weight ~96.9% missing
# max_glu_serum ~94.7% missing
# A1Cresult ~83.3% missing
HIGH_MISSING_COLUMNS = [
    "weight",
    "max_glu_serum",
    "A1Cresult",
]

# payer_code has substantial missingness and limited relevance to the
# clinical readmission objective, so it is excluded from the baseline model.
BASELINE_DROP_COLUMNS = [
    "payer_code",
]

DIAGNOSIS_COLUMNS = [
    "diag_1",
    "diag_2",
    "diag_3",
]

NUMERIC_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# These columns are treated as categorical even if some are stored as integers,
# because they represent codes rather than quantities.
CATEGORICAL_COLUMNS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "insulin",
    "change",
    "diabetesMed",
]

REQUIRED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + [ORIGINAL_TARGET_COLUMN]
    + DIAGNOSIS_COLUMNS
    + NUMERIC_COLUMNS
    + CATEGORICAL_COLUMNS
)


@dataclass
class DiabetesPreprocessingResult:
    """Container returned by prepare_diabetes_data."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    cleaned: pd.DataFrame
    numeric_columns: list[str]
    categorical_columns: list[str]


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    """Raise a clear error when expected source columns are missing."""

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            "The Diabetes dataset is missing required columns: "
            + ", ".join(missing)
        )


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataset-specific missing markers to standard NaN values."""

    cleaned = df.copy()
    cleaned = cleaned.replace("?", np.nan)
    return cleaned


def create_readmission_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary 30-day readmission target.

    Target definition:
        1 = original readmitted value is "<30"
        0 = original readmitted value is ">30" or "NO"
    """

    if ORIGINAL_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{ORIGINAL_TARGET_COLUMN}' is required to create the target."
        )

    result = df.copy()
    result[TARGET_COLUMN] = (
        result[ORIGINAL_TARGET_COLUMN].astype(str).eq("<30")
    ).astype("int8")

    return result


def drop_baseline_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns excluded from the baseline predictive model."""

    drop_columns = [
        *HIGH_MISSING_COLUMNS,
        *BASELINE_DROP_COLUMNS,
    ]

    existing = [column for column in drop_columns if column in df.columns]
    return df.drop(columns=existing)


def _parse_icd_numeric(value: object) -> float | None:
    """Convert a diagnosis code to its numeric component when possible.

    ICD values beginning with V or E are non-numeric supplemental/external
    cause codes and are returned as None for the numeric grouping logic.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text or text.upper().startswith(("V", "E")):
        return None

    try:
        return float(text)
    except ValueError:
        return None


def group_diagnosis_code(value: object) -> str:
    """Group raw diagnosis codes into broad, interpretable categories.

    The grouping is intentionally coarse to reduce the high cardinality of
    diag_1, diag_2, and diag_3 before model encoding.
    """

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().upper()

    if not text:
        return "Unknown"

    if text.startswith("V"):
        return "Supplemental_Factors"

    if text.startswith("E"):
        return "External_Causes"

    code = _parse_icd_numeric(text)

    if code is None:
        return "Other"

    if 390 <= code <= 459 or code == 785:
        return "Circulatory"

    if 460 <= code <= 519 or code == 786:
        return "Respiratory"

    if 520 <= code <= 579 or code == 787:
        return "Digestive"

    if 250 <= code < 251:
        return "Diabetes"

    if 800 <= code <= 999:
        return "Injury"

    if 710 <= code <= 739:
        return "Musculoskeletal"

    if 580 <= code <= 629 or code == 788:
        return "Genitourinary"

    if 140 <= code <= 239:
        return "Neoplasms"

    if 240 <= code <= 279:
        return "Endocrine_Metabolic"

    if 680 <= code <= 709 or code == 782:
        return "Skin"

    if 1 <= code <= 139:
        return "Infectious"

    if 290 <= code <= 319:
        return "Mental"

    if 320 <= code <= 389:
        return "Nervous_Sensory"

    if 630 <= code <= 679:
        return "Pregnancy"

    if 740 <= code <= 759:
        return "Congenital"

    if 760 <= code <= 779:
        return "Perinatal"

    if 780 <= code <= 799:
        return "Symptoms"

    return "Other"


def group_diagnosis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw diagnosis columns with grouped diagnosis features."""

    result = df.copy()

    for column in DIAGNOSIS_COLUMNS:
        if column in result.columns:
            grouped_column = f"{column}_group"
            result[grouped_column] = result[column].apply(group_diagnosis_code)

    existing_raw = [
        column for column in DIAGNOSIS_COLUMNS if column in result.columns
    ]

    result = result.drop(columns=existing_raw)

    return result


def clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize missing and rare invalid values in categorical features."""

    result = df.copy()

    # Convert code-like categorical variables to strings so later one-hot
    # encoding treats them as categories rather than continuous numbers.
    code_columns = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]

    for column in code_columns:
        if column in result.columns:
            result[column] = result[column].astype("string")

    categorical_candidates = result.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_candidates:
        result[column] = (
            result[column]
            .astype("string")
            .fillna("Unknown")
            .str.strip()
            .replace("", "Unknown")
        )

    return result


def remove_invalid_gender_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extremely rare invalid/unknown gender values.

    The source dataset can contain a very small number of rows labeled
    'Unknown/Invalid'. Excluding those rows avoids creating a category with
    almost no support in the baseline model.
    """

    if "gender" not in df.columns:
        return df.copy()

    result = df.copy()

    valid_mask = ~result["gender"].isin(["Unknown/Invalid"])
    return result.loc[valid_mask].copy()


def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply reusable cleaning and feature-engineering steps.

    This function does not one-hot encode or scale features. Those operations
    should be learned from training data inside a scikit-learn Pipeline or
    ColumnTransformer to prevent train/test leakage.
    """

    validate_required_columns(df)

    cleaned = normalize_missing_values(df)
    cleaned = create_readmission_target(cleaned)
    cleaned = drop_baseline_columns(cleaned)
    cleaned = remove_invalid_gender_rows(cleaned)
    cleaned = group_diagnosis_columns(cleaned)
    cleaned = clean_categorical_values(cleaned)

    return cleaned.reset_index(drop=True)


def prepare_diabetes_data(
    df: pd.DataFrame,
) -> DiabetesPreprocessingResult:
    """Prepare feature, target, and patient-group objects for modeling.

    Returns
    -------
    DiabetesPreprocessingResult
        X:
            Predictive feature matrix with identifiers and the original target
            removed.
        y:
            Binary 30-day readmission target.
        groups:
            patient_nbr values retained exclusively for leakage-aware splitting.
        cleaned:
            Cleaned dataframe before identifier removal.
        numeric_columns:
            Numerical columns available in X.
        categorical_columns:
            Categorical columns available in X.
    """

    cleaned = clean_diabetes_data(df)

    groups = cleaned["patient_nbr"].copy()
    y = cleaned[TARGET_COLUMN].copy()

    columns_to_remove = [
        column
        for column in [
            *IDENTIFIER_COLUMNS,
            ORIGINAL_TARGET_COLUMN,
            TARGET_COLUMN,
        ]
        if column in cleaned.columns
    ]

    X = cleaned.drop(columns=columns_to_remove)

    numeric_columns = [
        column
        for column in X.columns
        if pd.api.types.is_numeric_dtype(X[column])
    ]

    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    return DiabetesPreprocessingResult(
        X=X,
        y=y,
        groups=groups,
        cleaned=cleaned,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def preprocessing_summary(
    result: DiabetesPreprocessingResult,
) -> dict[str, object]:
    """Return a compact summary useful for notebooks and logging."""

    return {
        "rows": int(len(result.X)),
        "features": int(result.X.shape[1]),
        "positive_cases": int(result.y.sum()),
        "positive_rate_percent": round(float(result.y.mean() * 100), 2),
        "unique_patients": int(result.groups.nunique()),
        "numeric_features": len(result.numeric_columns),
        "categorical_features": len(result.categorical_columns),
        "remaining_missing_values": int(result.X.isna().sum().sum()),
    }


if __name__ == "__main__":
    # Lightweight smoke-test example.
    # Run from the project root with:
    # python -m src.preprocessing.diabetes
    from pathlib import Path

    data_file = (
        Path("data")
        / "raw"
        / "diabetes"
        / "diabetic_data.csv"
    )

    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find {data_file}. "
            "Place diabetic_data.csv in data/raw/diabetes/."
        )

    raw_df = pd.read_csv(data_file)
    prepared = prepare_diabetes_data(raw_df)

    print("Diabetes preprocessing completed successfully.")
    print(preprocessing_summary(prepared))
