from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.safeguards import validate_request


def test_safe_clinical_instruction_is_allowed():
    result = validate_request({
        "task": "clinical_risk",
        "clinical_record": {"example": 1},
        "message": "Explain the result. Do not diagnose or prescribe.",
    })

    assert result.allowed is True


def test_safe_integrated_instruction_is_allowed():
    result = validate_request({
        "task": "integrated_assessment",
        "clinical_record": {"example": 1},
        "sensor_window": [[0.0]],
        "message": (
            "Summarize independent signals. "
            "Do not diagnose, prescribe, or imply causation."
        ),
    })

    assert result.allowed is True


def test_diagnosis_request_is_blocked():
    result = validate_request({
        "task": "clinical_risk",
        "clinical_record": {"example": 1},
        "message": "Diagnose my condition.",
    })

    assert result.allowed is False


def test_prescribing_request_is_blocked():
    result = validate_request({
        "task": "clinical_risk",
        "clinical_record": {"example": 1},
        "message": "What medication should I take?",
    })

    assert result.allowed is False
