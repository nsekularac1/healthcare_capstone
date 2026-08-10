from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.diabetes import prepare_diabetes_data
from src.prediction.readmission import (
    load_readmission_model,
    predict_readmission,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "diabetes"
    / "diabetic_data.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "readmission_model.pkl"
)


def test_readmission_model_inference():
    raw_df = pd.read_csv(DATA_PATH)
    prepared = prepare_diabetes_data(raw_df)

    model = load_readmission_model(MODEL_PATH)

    sample = prepared.X.iloc[:3].copy()

    predictions = predict_readmission(
        model=model,
        X=sample,
        threshold=0.50,
    )

    assert len(predictions) == 3

    assert "readmission_probability" in predictions.columns
    assert "predicted_readmission" in predictions.columns

    assert predictions["readmission_probability"].between(0, 1).all()
    assert set(
        predictions["predicted_readmission"].unique()
    ).issubset({0, 1})
