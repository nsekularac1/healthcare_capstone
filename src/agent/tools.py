"""Local model tools used by the OpenAI-backed healthcare agent.

The OpenAI model decides which tool(s) to call. The actual predictive
computation remains local:

- `predict_readmission` uses the saved scikit-learn clinical pipeline.
- `classify_activity` uses the saved PyTorch PAMAP2 LSTM.

Raw clinical records and sensor arrays are NOT sent to OpenAI as tool
arguments. The agent only tells OpenAI which input types are available.
When OpenAI selects a tool, this module executes it against the local payload
already supplied to the application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.activity.lstm import (
    get_device,
    load_activity_model,
    predict_activity_probabilities,
)
from src.prediction.readmission import (
    load_readmission_model,
    predict_readmission,
)


def clinical_risk_tool(
    clinical_record: dict[str, Any],
    *,
    model_path: str | Path,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Run the saved 30-day readmission pipeline for one prepared encounter."""

    if not isinstance(clinical_record, dict) or not clinical_record:
        raise ValueError("clinical_record must be a non-empty dictionary.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    model = load_readmission_model(model_path)
    frame = pd.DataFrame([clinical_record])

    prediction = predict_readmission(
        model=model,
        X=frame,
        threshold=threshold,
    ).iloc[0]

    probability = float(prediction["readmission_probability"])
    predicted = int(prediction["predicted_readmission"])

    # Human-readable presentation tier. The model probability remains the
    # primary quantitative output.
    if probability >= 0.70:
        risk_level = "high"
    elif probability >= 0.40:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "readmission_probability": probability,
        "predicted_readmission": predicted,
        "risk_level": risk_level,
        "classification_threshold": float(threshold),
        "provenance": "Diabetes 130-US Hospitals clinical prediction pipeline",
    }


def activity_recognition_tool(
    sensor_window: np.ndarray | list,
    *,
    model_path: str | Path,
) -> dict[str, Any]:
    """Classify one already-preprocessed/scaled PAMAP2 sensor window."""

    X = np.asarray(sensor_window, dtype=np.float32)

    if X.ndim == 2:
        X = X[np.newaxis, ...]

    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError(
            "sensor_window must have shape (timesteps, features) "
            "or (1, timesteps, features)."
        )

    if np.isnan(X).any():
        raise ValueError("sensor_window contains NaN values.")

    device = get_device()
    model, metadata = load_activity_model(
        model_path,
        device=device,
    )

    expected_window = int(metadata["window_size"])
    expected_features = int(metadata["input_size"])

    if X.shape[1] != expected_window:
        raise ValueError(
            f"Expected {expected_window} time steps, received {X.shape[1]}."
        )

    if X.shape[2] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, received {X.shape[2]}."
        )

    probabilities = predict_activity_probabilities(
        model=model,
        X=X,
        device=device,
    )[0]

    predicted_index = int(probabilities.argmax())
    confidence = float(probabilities[predicted_index])
    class_names = list(metadata["class_names"])

    return {
        "predicted_activity": class_names[predicted_index],
        "confidence": confidence,
        "class_index": predicted_index,
        "provenance": "PAMAP2 wearable activity-recognition pipeline",
    }


def execute_agent_tool(
    tool_name: str,
    *,
    clinical_record: dict[str, Any] | None,
    sensor_window: np.ndarray | list | None,
    readmission_model_path: str | Path,
    activity_model_path: str | Path,
    readmission_threshold: float,
) -> dict[str, Any]:
    """Dispatch an OpenAI-selected tool to the appropriate local function."""

    if tool_name == "predict_readmission":
        if clinical_record is None:
            raise ValueError(
                "OpenAI selected predict_readmission but no clinical_record "
                "was supplied."
            )

        return clinical_risk_tool(
            clinical_record,
            model_path=readmission_model_path,
            threshold=readmission_threshold,
        )

    if tool_name == "classify_activity":
        if sensor_window is None:
            raise ValueError(
                "OpenAI selected classify_activity but no sensor_window "
                "was supplied."
            )

        return activity_recognition_tool(
            sensor_window,
            model_path=activity_model_path,
        )

    raise ValueError(f"Unsupported agent tool: {tool_name}")
