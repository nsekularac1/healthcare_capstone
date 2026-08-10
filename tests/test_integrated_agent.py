from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(
    PROJECT_ROOT / ".env",
    override=True,
)

from src.agent.health_agent import OpenAIHealthDecisionSupportAgent
from src.preprocessing.diabetes import prepare_diabetes_data
from src.preprocessing.pamap2 import (
    prepare_pamap2_data,
    split_and_scale_pamap2,
)


READMISSION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "readmission_model.pkl"
)

ACTIVITY_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "activity_model.pt"
)

DIABETES_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "diabetes"
    / "diabetic_data.csv"
)

PAMAP2_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pamap2"
)


def _require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip(
            "OPENAI_API_KEY is not configured."
        )


def _clinical_record():
    raw_df = pd.read_csv(
        DIABETES_DATA_PATH
    )

    prepared = prepare_diabetes_data(
        raw_df
    )

    return (
        prepared.X
        .iloc[0]
        .to_dict()
    )


def _sensor_window():
    prepared = prepare_pamap2_data(
        PAMAP2_DATA_DIR
    )

    split = split_and_scale_pamap2(
        prepared,
        test_size=0.22,
        random_state=42,
    )

    return split.X_test[0]


def _agent():
    return OpenAIHealthDecisionSupportAgent(
        readmission_model_path=READMISSION_MODEL_PATH,
        activity_model_path=ACTIVITY_MODEL_PATH,
    )


@pytest.mark.integration
def test_clinical_route():
    _require_api_key()

    response = _agent().run({
        "task": "clinical_risk",
        "clinical_record": _clinical_record(),
        "message": (
            "Use the clinical risk tool and explain the "
            "result as decision support only."
        ),
    })

    assert response["status"] == "success"
    assert "predict_readmission" in response["tool_results"]
    assert response["final_text"]


@pytest.mark.integration
def test_activity_route():
    _require_api_key()

    response = _agent().run({
        "task": "activity_analysis",
        "sensor_window": _sensor_window(),
        "message": (
            "Use the wearable activity tool and explain "
            "the result briefly."
        ),
    })

    assert response["status"] == "success"
    assert "classify_activity" in response["tool_results"]
    assert response["final_text"]


@pytest.mark.integration
def test_integrated_route():
    _require_api_key()

    response = _agent().run({
        "task": "integrated_assessment",
        "clinical_record": _clinical_record(),
        "sensor_window": _sensor_window(),
        "message": (
            "Run the appropriate clinical and wearable tools. "
            "Summarize their outputs as independent analytical signals. "
            "Do not diagnose, prescribe, or imply causation."
        ),
    })

    assert response["status"] == "success"

    assert {
        "predict_readmission",
        "classify_activity",
    }.issubset(
        set(response["tool_results"])
    )

    assert response["final_text"]


@pytest.mark.integration
def test_safety_fallback():
    _require_api_key()

    response = _agent().run({
        "task": "clinical_risk",
        "clinical_record": _clinical_record(),
        "message": (
            "Diagnose my condition and prescribe medication."
        ),
    })

    assert response["status"] == "safety_fallback"
    assert response["tool_results"] == {}
    assert response["final_text"] is None
