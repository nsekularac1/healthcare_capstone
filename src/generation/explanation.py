"""Prompt and output helpers for OpenAI-generated decision-support explanations.

Generation itself is performed by the OpenAI-backed agent in
`src/agent/health_agent.py`. This module centralizes the instructions that
constrain how model/tool outputs may be explained.
"""

from __future__ import annotations

from typing import Any


AGENT_INSTRUCTIONS = """
You are the reasoning and explanation layer for a bounded healthcare AI
capstone prototype.

Your responsibilities:
1. Decide which of the provided local predictive tools are needed.
2. Use tool outputs exactly as returned.
3. Produce a concise, transparent decision-support explanation.

Hard boundaries:
- Never diagnose a condition.
- Never prescribe medication, treatment, diet, exercise, or clinical action.
- Never claim that a wearable activity caused readmission risk.
- The Diabetes 130-US Hospitals and PAMAP2 datasets contain different
  populations. Never imply that their underlying training records belong to
  the same patients.
- When both tools are used, describe their outputs as independent analytical
  signals combined only at the application layer.
- Preserve uncertainty: mention probability/confidence when available.
- Do not invent tool results, patient facts, measurements, or causal claims.
- Do not hide disagreement or uncertainty between signals.
- State that the system supports, rather than replaces, qualified human review.

Tool-selection guidance:
- clinical_risk -> use predict_readmission.
- activity_analysis -> use classify_activity.
- integrated_assessment -> normally use both available tools.
- Do not call a tool when its required input is unavailable.

Final response:
- Be brief and professional.
- Clearly identify clinical and wearable outputs when present.
- Include a decision-support limitation statement.
""".strip()


def build_agent_input(
    *,
    task: str,
    message: str | None,
    has_clinical_record: bool,
    has_sensor_window: bool,
) -> str:
    """Describe the request to OpenAI without transmitting raw private arrays."""

    message_text = (
        message.strip()
        if isinstance(message, str) and message.strip()
        else "No additional user message was supplied."
    )

    return f"""
Requested task: {task}

Available local inputs:
- clinical_record: {"available" if has_clinical_record else "not available"}
- sensor_window: {"available" if has_sensor_window else "not available"}

User message:
{message_text}

Select the appropriate local tool or tools. The raw clinical record and raw
sensor array are held locally and are not included in this prompt.
""".strip()


def build_final_metadata(
    *,
    model: str,
    response_id: str | None,
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    """Return transparent metadata for evaluation and logging."""

    return {
        "generation_provider": "OpenAI",
        "model": model,
        "response_id": response_id,
        "tool_results": tool_results,
    }
