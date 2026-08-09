"""Reusable preprocessing utilities for the PAMAP2 activity-recognition branch.

This module converts raw PAMAP2 subject files into fixed-length sensor
sequences suitable for an LSTM activity-recognition model.

Design goals
------------
- Load all PAMAP2 `subject*.dat` files with explicit column names.
- Remove transient/unlabeled activity ID 0.
- Retain a focused set of heart-rate, accelerometer, and gyroscope channels.
- Handle missing sensor values within subject/activity segments.
- Create fixed-length overlapping windows that never cross subject or
  activity boundaries.
- Encode activity labels consistently.
- Support subject-aware train/test splitting.
- Fit sensor scaling on TRAINING data only to avoid leakage.

The module intentionally keeps model definition and training separate.
Those responsibilities belong in `src/activity/lstm.py` and
`04_activity_model.ipynb`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

WINDOW_SIZE = 100
WINDOW_STEP = 50

ACTIVITY_LABELS = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

IMU_FEATURES = [
    "temperature",
    "acc16_x",
    "acc16_y",
    "acc16_z",
    "acc6_x",
    "acc6_y",
    "acc6_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "orientation_1",
    "orientation_2",
    "orientation_3",
    "orientation_4",
]

COLUMN_NAMES = [
    "timestamp",
    "activity_id",
    "heart_rate",
]

for location in ["hand", "chest", "ankle"]:
    COLUMN_NAMES.extend(
        f"{location}_{feature}"
        for feature in IMU_FEATURES
    )

SELECTED_SENSOR_FEATURES = [
    "heart_rate",

    "hand_acc16_x",
    "hand_acc16_y",
    "hand_acc16_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",

    "chest_acc16_x",
    "chest_acc16_y",
    "chest_acc16_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",

    "ankle_acc16_x",
    "ankle_acc16_y",
    "ankle_acc16_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
]

REQUIRED_METADATA_COLUMNS = [
    "timestamp",
    "activity_id",
    "subject_id",
]


# ---------------------------------------------------------------------------
# Returned data structures
# ---------------------------------------------------------------------------

@dataclass
class PAMAP2PreprocessingResult:
    """Windowed PAMAP2 data before train/test scaling."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray

    feature_names: list[str]
    class_names: list[str]

    label_encoder: LabelEncoder

    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame


@dataclass
class PAMAP2SplitResult:
    """Subject-separated and train-fitted scaled PAMAP2 arrays."""

    X_train: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_test: np.ndarray

    groups_train: np.ndarray
    groups_test: np.ndarray

    train_indices: np.ndarray
    test_indices: np.ndarray

    scaler: StandardScaler


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_window_parameters(
    window_size: int,
    window_step: int,
) -> None:
    """Validate sequence-window configuration."""

    if window_size <= 0:
        raise ValueError("window_size must be greater than zero.")

    if window_step <= 0:
        raise ValueError("window_step must be greater than zero.")

    if window_step > window_size:
        raise ValueError(
            "window_step should not exceed window_size for this project."
        )


def validate_selected_features(
    df: pd.DataFrame,
    selected_features: Iterable[str],
) -> None:
    """Ensure all requested sensor features exist."""

    missing = [
        feature
        for feature in selected_features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "PAMAP2 data is missing selected features: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def find_subject_files(
    data_dir: str | Path,
) -> list[Path]:
    """Find PAMAP2 subject files recursively."""

    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"PAMAP2 data directory does not exist: {root}"
        )

    files = sorted(root.rglob("subject*.dat"))

    if not files:
        raise FileNotFoundError(
            f"No subject*.dat files were found under: {root}"
        )

    return files


def load_subject_file(
    file_path: str | Path,
) -> pd.DataFrame:
    """Load one PAMAP2 subject protocol file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PAMAP2 subject file does not exist: {path}"
        )

    subject_text = path.stem.replace("subject", "")

    try:
        subject_id = int(subject_text)
    except ValueError:
        subject_id = subject_text

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        na_values="NaN",
    )

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Unexpected column count in {path.name}: "
            f"{df.shape[1]} instead of {len(COLUMN_NAMES)}."
        )

    df["subject_id"] = subject_id
    return df


def load_pamap2_data(
    data_dir: str | Path,
) -> pd.DataFrame:
    """Load and concatenate all PAMAP2 subject files."""

    subject_files = find_subject_files(data_dir)

    frames = [
        load_subject_file(file_path)
        for file_path in subject_files
    ]

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def remove_transient_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove activity ID 0, which represents transient/unlabeled periods."""

    if "activity_id" not in df.columns:
        raise ValueError(
            "Column 'activity_id' is required."
        )

    result = df.loc[
        df["activity_id"] != 0
    ].copy()

    return result.reset_index(drop=True)


def add_activity_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add human-readable activity names."""

    if "activity_id" not in df.columns:
        raise ValueError(
            "Column 'activity_id' is required."
        )

    result = df.copy()

    result["activity"] = (
        result["activity_id"]
        .map(ACTIVITY_LABELS)
        .fillna("unknown")
    )

    return result


def select_modeling_columns(
    df: pd.DataFrame,
    selected_features: Iterable[str] = SELECTED_SENSOR_FEATURES,
) -> pd.DataFrame:
    """Retain metadata plus the selected LSTM sensor channels."""

    selected_features = list(selected_features)
    validate_selected_features(
        df,
        selected_features,
    )

    required = [
        column
        for column in REQUIRED_METADATA_COLUMNS
        if column in df.columns
    ]

    if "activity" in df.columns:
        required.append("activity")

    return df[
        required + selected_features
    ].copy()


def remove_duplicate_sensor_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove exact duplicate rows."""

    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------

def interpolate_sensor_values(
    df: pd.DataFrame,
    selected_features: Iterable[str] = SELECTED_SENSOR_FEATURES,
) -> pd.DataFrame:
    """Fill missing sensor values without crossing subject/activity boundaries.

    Processing is performed independently for each subject/activity segment.

    Steps within each segment:
    1. sort by timestamp;
    2. linearly interpolate internal missing values;
    3. forward-fill and backward-fill edge values;
    4. use a within-subject median fallback if an entire segment lacks a
       feature;
    5. use the dataset median only as a final fallback.

    Notes
    -----
    For model evaluation, scaling is handled separately and fitted only on
    training windows. The interpolation fallback medians here are used solely
    to resolve missing raw sensor measurements.
    """

    selected_features = list(selected_features)
    validate_selected_features(
        df,
        selected_features,
    )

    result = df.copy()

    # Pre-compute fallback medians.
    subject_medians = result.groupby("subject_id")[
        selected_features
    ].median(numeric_only=True)

    dataset_medians = result[
        selected_features
    ].median(numeric_only=True)

    processed_segments = []

    for (subject_id, activity_id), segment in result.groupby(
        ["subject_id", "activity_id"],
        sort=False,
    ):
        segment = segment.sort_values(
            "timestamp"
        ).copy()

        # Interpolate independently inside the activity segment.
        segment[selected_features] = (
            segment[selected_features]
            .interpolate(
                method="linear",
                limit_direction="both",
            )
            .ffill()
            .bfill()
        )

        # Segment can still contain NaN if a complete channel is absent.
        if segment[selected_features].isna().any().any():
            if subject_id in subject_medians.index:
                segment[selected_features] = (
                    segment[selected_features]
                    .fillna(
                        subject_medians.loc[subject_id]
                    )
                )

        segment[selected_features] = (
            segment[selected_features]
            .fillna(dataset_medians)
        )

        processed_segments.append(segment)

    cleaned = pd.concat(
        processed_segments,
        ignore_index=True,
    )

    # If any column remains entirely missing, fail clearly rather than
    # silently producing invalid LSTM inputs.
    remaining = cleaned[
        selected_features
    ].isna().sum()

    unresolved = remaining[
        remaining > 0
    ]

    if not unresolved.empty:
        raise ValueError(
            "Missing sensor values remain after interpolation: "
            + ", ".join(
                f"{column}={count}"
                for column, count in unresolved.items()
            )
        )

    return cleaned


# ---------------------------------------------------------------------------
# Activity segments and window generation
# ---------------------------------------------------------------------------

def add_activity_segment_ids(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify contiguous activity segments within each participant.

    A new segment begins whenever:
    - the subject changes; or
    - the activity ID changes.

    This prevents sequence windows from crossing activity transitions.
    """

    required = {
        "subject_id",
        "activity_id",
        "timestamp",
    }

    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(
            "Missing columns required for segmentation: "
            + ", ".join(sorted(missing))
        )

    result = df.sort_values(
        ["subject_id", "timestamp"]
    ).copy()

    subject_change = (
        result["subject_id"]
        .ne(result["subject_id"].shift())
    )

    activity_change = (
        result["activity_id"]
        .ne(result["activity_id"].shift())
    )

    result["segment_id"] = (
        subject_change | activity_change
    ).cumsum()

    return result.reset_index(drop=True)


def create_activity_windows(
    df: pd.DataFrame,
    selected_features: Iterable[str] = SELECTED_SENSOR_FEATURES,
    *,
    window_size: int = WINDOW_SIZE,
    window_step: int = WINDOW_STEP,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Convert continuous sensor segments into overlapping LSTM windows.

    Windows never cross participant or contiguous activity boundaries.

    Returns
    -------
    X:
        Shape `(n_windows, window_size, n_features)`.

    activity_ids:
        Integer PAMAP2 activity ID for each window.

    groups:
        Subject identifier for each window.
    """

    validate_window_parameters(
        window_size,
        window_step,
    )

    selected_features = list(
        selected_features
    )

    validate_selected_features(
        df,
        selected_features,
    )

    if "segment_id" not in df.columns:
        working = add_activity_segment_ids(df)
    else:
        working = df.copy()

    X_windows: list[np.ndarray] = []
    y_windows: list[int] = []
    group_windows: list[object] = []

    for _, segment in working.groupby(
        "segment_id",
        sort=False,
    ):
        segment = segment.sort_values(
            "timestamp"
        )

        n_rows = len(segment)

        if n_rows < window_size:
            continue

        activity_values = (
            segment["activity_id"]
            .dropna()
            .unique()
        )

        subject_values = (
            segment["subject_id"]
            .dropna()
            .unique()
        )

        # Segmentation logic should guarantee exactly one activity and subject.
        if len(activity_values) != 1:
            raise RuntimeError(
                "A sequence segment contains multiple activity IDs."
            )

        if len(subject_values) != 1:
            raise RuntimeError(
                "A sequence segment contains multiple subjects."
            )

        activity_id = int(
            activity_values[0]
        )

        subject_id = subject_values[0]

        feature_array = segment[
            selected_features
        ].to_numpy(
            dtype=np.float32
        )

        for start in range(
            0,
            n_rows - window_size + 1,
            window_step,
        ):
            stop = start + window_size

            window = feature_array[
                start:stop
            ]

            if window.shape[0] != window_size:
                continue

            if np.isnan(window).any():
                raise ValueError(
                    "NaN detected while creating activity windows."
                )

            X_windows.append(window)
            y_windows.append(activity_id)
            group_windows.append(subject_id)

    if not X_windows:
        raise ValueError(
            "No PAMAP2 windows were created. "
            "Check window parameters and input data."
        )

    X = np.stack(
        X_windows
    ).astype(np.float32)

    activity_ids = np.asarray(
        y_windows,
        dtype=np.int64,
    )

    groups = np.asarray(
        group_windows
    )

    return X, activity_ids, groups


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_activity_labels(
    activity_ids: np.ndarray,
) -> tuple[
    np.ndarray,
    LabelEncoder,
    list[str],
]:
    """Encode PAMAP2 activity IDs as contiguous class indices."""

    activity_ids = np.asarray(
        activity_ids
    )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        activity_ids
    ).astype(np.int64)

    class_names = [
        ACTIVITY_LABELS.get(
            int(activity_id),
            str(activity_id),
        )
        for activity_id in label_encoder.classes_
    ]

    return (
        y,
        label_encoder,
        class_names,
    )


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def prepare_pamap2_data(
    data_dir: str | Path,
    *,
    selected_features: Iterable[str] = SELECTED_SENSOR_FEATURES,
    window_size: int = WINDOW_SIZE,
    window_step: int = WINDOW_STEP,
) -> PAMAP2PreprocessingResult:
    """Load, clean, interpolate, segment, and window PAMAP2 data.

    Scaling is intentionally NOT performed here. Scaling must be fitted on
    training windows only after the participant-aware train/test split.
    """

    selected_features = list(
        selected_features
    )

    raw = load_pamap2_data(
        data_dir
    )

    cleaned = remove_transient_rows(
        raw
    )

    cleaned = add_activity_names(
        cleaned
    )

    cleaned = select_modeling_columns(
        cleaned,
        selected_features,
    )

    cleaned = remove_duplicate_sensor_rows(
        cleaned
    )

    cleaned = interpolate_sensor_values(
        cleaned,
        selected_features,
    )

    cleaned = add_activity_segment_ids(
        cleaned
    )

    X, activity_ids, groups = (
        create_activity_windows(
            cleaned,
            selected_features,
            window_size=window_size,
            window_step=window_step,
        )
    )

    y, label_encoder, class_names = (
        encode_activity_labels(
            activity_ids
        )
    )

    return PAMAP2PreprocessingResult(
        X=X,
        y=y,
        groups=groups,
        feature_names=selected_features,
        class_names=class_names,
        label_encoder=label_encoder,
        raw_data=raw,
        cleaned_data=cleaned,
    )


# ---------------------------------------------------------------------------
# Subject-aware splitting and scaling
# ---------------------------------------------------------------------------

def subject_aware_split(
    result: PAMAP2PreprocessingResult,
    *,
    test_size: float = 0.22,
    random_state: int = RANDOM_STATE,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Create train/test indices with complete subject separation."""

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(
            result.X,
            result.y,
            groups=result.groups,
        )
    )

    train_subjects = set(
        result.groups[train_idx]
    )

    test_subjects = set(
        result.groups[test_idx]
    )

    overlap = train_subjects.intersection(
        test_subjects
    )

    if overlap:
        raise RuntimeError(
            "Subject leakage detected: "
            + ", ".join(
                map(str, sorted(overlap))
            )
        )

    return train_idx, test_idx


def fit_sequence_scaler(
    X_train: np.ndarray,
) -> StandardScaler:
    """Fit StandardScaler using training sensor values only."""

    if X_train.ndim != 3:
        raise ValueError(
            "X_train must have shape "
            "(samples, timesteps, features)."
        )

    n_features = X_train.shape[2]

    flattened = X_train.reshape(
        -1,
        n_features,
    )

    scaler = StandardScaler()
    scaler.fit(flattened)

    return scaler


def transform_sequences(
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply a fitted feature scaler to 3-D sequence data."""

    if X.ndim != 3:
        raise ValueError(
            "X must have shape "
            "(samples, timesteps, features)."
        )

    original_shape = X.shape

    flattened = X.reshape(
        -1,
        original_shape[2],
    )

    transformed = scaler.transform(
        flattened
    )

    return transformed.reshape(
        original_shape
    ).astype(np.float32)


def split_and_scale_pamap2(
    result: PAMAP2PreprocessingResult,
    *,
    test_size: float = 0.22,
    random_state: int = RANDOM_STATE,
) -> PAMAP2SplitResult:
    """Perform participant-aware splitting and leakage-safe scaling."""

    train_idx, test_idx = (
        subject_aware_split(
            result,
            test_size=test_size,
            random_state=random_state,
        )
    )

    X_train_raw = result.X[
        train_idx
    ]

    X_test_raw = result.X[
        test_idx
    ]

    scaler = fit_sequence_scaler(
        X_train_raw
    )

    X_train = transform_sequences(
        X_train_raw,
        scaler,
    )

    X_test = transform_sequences(
        X_test_raw,
        scaler,
    )

    return PAMAP2SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=result.y[train_idx],
        y_test=result.y[test_idx],
        groups_train=result.groups[train_idx],
        groups_test=result.groups[test_idx],
        train_indices=train_idx,
        test_indices=test_idx,
        scaler=scaler,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def preprocessing_summary(
    result: PAMAP2PreprocessingResult,
) -> dict[str, object]:
    """Return a compact preprocessing summary."""

    class_counts = pd.Series(
        result.y
    ).value_counts().sort_index()

    return {
        "raw_rows": int(
            len(result.raw_data)
        ),
        "cleaned_rows": int(
            len(result.cleaned_data)
        ),
        "windows": int(
            result.X.shape[0]
        ),
        "window_size": int(
            result.X.shape[1]
        ),
        "features": int(
            result.X.shape[2]
        ),
        "subjects": int(
            pd.Series(
                result.groups
            ).nunique()
        ),
        "classes": int(
            len(result.class_names)
        ),
        "class_names": result.class_names,
        "class_counts": {
            result.class_names[index]: int(
                class_counts.get(index, 0)
            )
            for index in range(
                len(result.class_names)
            )
        },
        "remaining_nan_values": int(
            np.isnan(result.X).sum()
        ),
    }


def split_summary(
    split: PAMAP2SplitResult,
) -> dict[str, object]:
    """Return participant and sequence counts for a split."""

    train_subjects = sorted(
        set(split.groups_train)
    )

    test_subjects = sorted(
        set(split.groups_test)
    )

    return {
        "training_windows": int(
            len(split.X_train)
        ),
        "testing_windows": int(
            len(split.X_test)
        ),
        "training_subjects": train_subjects,
        "testing_subjects": test_subjects,
        "subject_overlap": sorted(
            set(train_subjects).intersection(
                test_subjects
            )
        ),
        "training_nan_values": int(
            np.isnan(
                split.X_train
            ).sum()
        ),
        "testing_nan_values": int(
            np.isnan(
                split.X_test
            ).sum()
        ),
    }


# ---------------------------------------------------------------------------
# Optional command-line smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(
        __file__
    ).resolve().parents[2]

    data_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "pamap2"
    )

    prepared = prepare_pamap2_data(
        data_dir
    )

    print(
        "PAMAP2 preprocessing completed successfully."
    )

    print(
        preprocessing_summary(
            prepared
        )
    )

    split = split_and_scale_pamap2(
        prepared
    )

    print(
        "Participant-aware split completed successfully."
    )

    print(
        split_summary(
            split
        )
    )
