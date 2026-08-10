"""Deterministic safety and scope controls for the OpenAI-backed healthcare agent.

The safeguards block explicit requests for diagnosis, prescribing, dosage, or
treatment selection while allowing safety instructions such as:

- "Do not diagnose."
- "Do not prescribe medication."
- "Do not diagnose, prescribe, or imply causation."

The screening intentionally examines only the free-text `message` field.
Structured clinical features and model payloads are not searched for trigger
words.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


SUPPORTED_TASKS = {
    "clinical_risk",
    "activity_analysis",
    "integrated_assessment",
}


@dataclass
class SafetyResult:
    """Result returned by deterministic medical-scope screening."""

    allowed: bool
    reason: str | None = None
    matched_terms: list[str] | None = None


def validate_request_structure(request: dict[str, Any]) -> None:
    """Validate supported tasks and required local payloads."""

    if not isinstance(request, dict):
        raise TypeError("request must be a dictionary.")

    task = request.get("task")

    if task not in SUPPORTED_TASKS:
        raise ValueError(
            "Unsupported task. Expected one of: "
            + ", ".join(sorted(SUPPORTED_TASKS))
        )

    if task == "clinical_risk":
        if not request.get("clinical_record"):
            raise ValueError(
                "clinical_risk requires clinical_record."
            )

    elif task == "activity_analysis":
        if request.get("sensor_window") is None:
            raise ValueError(
                "activity_analysis requires sensor_window."
            )

    elif task == "integrated_assessment":
        if not request.get("clinical_record"):
            raise ValueError(
                "integrated_assessment requires clinical_record."
            )

        if request.get("sensor_window") is None:
            raise ValueError(
                "integrated_assessment requires sensor_window."
            )


def _normalize_message(message: str) -> str:
    """Normalize message text for rule-based screening."""

    return re.sub(
        r"\s+",
        " ",
        message.lower().strip(),
    )


def _is_safety_constraint(text: str) -> bool:
    """Return True when the message is explicitly constraining unsafe behavior.

    This catches common forms used by this project, including coordinated
    phrases such as "do not diagnose, prescribe, or imply causation."
    """

    safety_patterns = [
        r"\bdo not\b.*\bdiagnos\w*\b",
        r"\bdo not\b.*\bprescrib\w*\b",
        r"\bdon't\b.*\bdiagnos\w*\b",
        r"\bdon't\b.*\bprescrib\w*\b",
        r"\bwithout\b.*\bdiagnos\w*\b",
        r"\bwithout\b.*\bprescrib\w*\b",
        r"\bnot a diagnosis\b",
        r"\bnot diagnostic\b",
        r"\bdecision support only\b",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in safety_patterns
    )


def screen_medical_scope(
    request: dict[str, Any],
) -> SafetyResult:
    """Block explicit requests for diagnosis, medication, dose, or treatment.

    The matcher uses request-like phrases rather than isolated words. This
    avoids false positives when "diagnose" or "prescribe" appears inside an
    instruction telling the system *not* to do those things.
    """

    raw_message = request.get("message", "")

    if raw_message is None:
        raw_message = ""

    if not isinstance(raw_message, str):
        raw_message = str(raw_message)

    text = _normalize_message(raw_message)

    # Explicitly safe constraints should not be blocked merely because they
    # contain words such as "diagnose" or "prescribe".
    safety_constraint = _is_safety_constraint(text)

    unsafe_patterns = {
        "diagnosis": [
            r"\bdiagnose me\b",
            r"\bdiagnose my\b",
            r"\bplease diagnose\b",
            r"\bcan you diagnose\b",
            r"\bcould you diagnose\b",
            r"\bwhat (?:condition|disease|illness) do i have\b",
            r"\btell me (?:my|the) diagnosis\b",
            r"\bgive me (?:a )?diagnosis\b",
        ],
        "prescribing": [
            r"\bprescribe (?:me )?(?:a |an |some )?\w+",
            r"\bplease prescribe\b",
            r"\bcan you prescribe\b",
            r"\bcould you prescribe\b",
            r"\bwhat medication should i take\b",
            r"\bwhich medication should i take\b",
            r"\bwhat drug should i take\b",
            r"\bwhich drug should i take\b",
        ],
        "dosage": [
            r"\bwhat dose should i take\b",
            r"\bwhat dosage should i take\b",
            r"\bhow much (?:of )?.* should i take\b",
            r"\btell me the dose\b",
            r"\btell me the dosage\b",
        ],
        "treatment selection": [
            r"\bwhat treatment should i use\b",
            r"\bwhich treatment should i use\b",
            r"\bgive me a treatment plan\b",
            r"\btell me how to treat\b",
            r"\bhow should i treat\b",
        ],
    }

    matched: list[str] = []

    for category, patterns in unsafe_patterns.items():
        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        ):
            matched.append(category)

    # A safety constraint such as "do not diagnose, prescribe, or imply
    # causation" can still technically match a broad phrase. If the message
    # is clearly phrased as a prohibition and does not independently ask for
    # diagnosis/treatment, allow it.
    if safety_constraint:
        direct_request_markers = [
            r"\bdiagnose me\b",
            r"\bdiagnose my\b",
            r"\bwhat medication should i take\b",
            r"\bwhich medication should i take\b",
            r"\bwhat dose should i take\b",
            r"\bwhat treatment should i use\b",
            r"\bgive me a treatment plan\b",
        ]

        has_direct_unsafe_request = any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in direct_request_markers
        )

        if not has_direct_unsafe_request:
            matched = []

    if matched:
        return SafetyResult(
            allowed=False,
            reason=(
                "The request exceeds this prototype's decision-support scope. "
                "Diagnosis, prescribing, dosage, and treatment selection are "
                "not supported."
            ),
            matched_terms=matched,
        )

    return SafetyResult(
        allowed=True,
        reason=None,
        matched_terms=[],
    )


def validate_request(
    request: dict[str, Any],
) -> SafetyResult:
    """Run structural validation and deterministic scope screening."""

    validate_request_structure(request)
    return screen_medical_scope(request)


if __name__ == "__main__":
    examples = [
        (
            "safe integrated instruction",
            {
                "task": "integrated_assessment",
                "clinical_record": {"example": 1},
                "sensor_window": [[0.0]],
                "message": (
                    "Run the appropriate clinical and wearable tools. "
                    "Summarize their outputs as independent analytical signals. "
                    "Do not diagnose, prescribe, or imply causation."
                ),
            },
        ),
        (
            "safe clinical instruction",
            {
                "task": "clinical_risk",
                "clinical_record": {"example": 1},
                "message": (
                    "Explain the risk result. Do not diagnose or prescribe."
                ),
            },
        ),
        (
            "unsafe diagnosis request",
            {
                "task": "clinical_risk",
                "clinical_record": {"example": 1},
                "message": (
                    "Diagnose my condition and prescribe medication for me."
                ),
            },
        ),
        (
            "unsafe medication request",
            {
                "task": "clinical_risk",
                "clinical_record": {"example": 1},
                "message": "What medication should I take?",
            },
        ),
    ]

    for name, example in examples:
        print(name)
        print(validate_request(example))
        print()
