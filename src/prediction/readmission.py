"""Reusable modeling utilities for 30-day diabetes readmission prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_RANDOM_STATE = 42


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Create the preprocessing transformer for numerical and categorical data."""

    if not numeric_columns and not categorical_columns:
        raise ValueError("At least one feature column is required.")

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_readmission_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 1000,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build the complete preprocessing + Logistic Regression pipeline."""

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    classifier = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="liblinear",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_readmission_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the readmission pipeline using training data only."""

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    if len(X_train) == 0:
        raise ValueError("Training data cannot be empty.")

    return model.fit(X_train, y_train)


def predict_readmission(
    model: Pipeline,
    X: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return readmission probabilities and binary predictions."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "readmission_probability": probabilities,
            "predicted_readmission": predictions,
        },
        index=X.index,
    )


def evaluate_readmission_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate accuracy, precision, recall, F1, ROC-AUC, and confusion matrix."""

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of rows.")

    predictions = predict_readmission(
        model=model,
        X=X_test,
        threshold=threshold,
    )

    y_pred = predictions["predicted_readmission"].to_numpy()
    y_prob = predictions["readmission_probability"].to_numpy()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "threshold": float(threshold),
        "test_rows": int(len(y_test)),
        "positive_cases": int(np.asarray(y_test).sum()),
    }


def extract_logistic_coefficients(
    model: Pipeline,
) -> pd.DataFrame:
    """Return transformed feature names and Logistic Regression coefficients."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    if len(feature_names) != len(coefficients):
        raise RuntimeError("Feature and coefficient counts do not match.")

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    result["absolute_coefficient"] = result["coefficient"].abs()

    return result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def save_readmission_model(
    model: Pipeline,
    path: str | Path,
) -> Path:
    """Save a fitted sklearn pipeline to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_readmission_model(
    path: str | Path,
) -> Pipeline:
    """Load a serialized sklearn readmission pipeline."""

    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Readmission model not found at: {model_path}"
        )

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError("Loaded object is not an sklearn Pipeline.")

    return model


def format_evaluation_metrics(
    metrics: dict[str, Any],
) -> pd.DataFrame:
    """Convert scalar evaluation metrics into a compact dataframe."""

    ordered_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    return pd.DataFrame(
        [
            {"metric": metric, "value": metrics[metric]}
            for metric in ordered_metrics
            if metric in metrics
        ]
    )
