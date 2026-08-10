from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.activity.lstm import (
    build_activity_model,
    load_activity_model,
    predict_activity_probabilities,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "activity_model.pt"
)


def test_lstm_forward_shape():
    model = build_activity_model(
        input_size=19,
        num_classes=12,
        hidden_size=64,
        num_layers=1,
    )

    X = torch.randn(
        4,
        100,
        19,
    )

    output = model(X)

    assert output.shape == (4, 12)


def test_saved_activity_model_inference():
    model, metadata = load_activity_model(MODEL_PATH)

    X = np.zeros(
        (2, metadata["window_size"], metadata["input_size"]),
        dtype=np.float32,
    )

    probabilities = predict_activity_probabilities(
        model,
        X,
    )

    assert probabilities.shape == (
        2,
        metadata["num_classes"],
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5,
    )
