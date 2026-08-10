"""OpenAI-backed healthcare decision-support agent with local function tools.

OpenAI performs:
- task reasoning;
- local tool selection through the Responses API function-calling mechanism;
- final natural-language explanation generation.

Local code performs:
- deterministic validation and safety screening;
- clinical readmission inference;
- PAMAP2 activity-recognition inference.

Raw clinical records and sensor windows remain local. OpenAI receives only
information about which local inputs are available plus the structured outputs
returned by the local tools.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.safeguards import validate_request
from src.agent.tools import execute_agent_tool
from src.generation.explanation import (
    AGENT_INSTRUCTIONS,
    build_agent_input,
    build_final_metadata,
)


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

if not ENV_FILE.exists():
    raise FileNotFoundError(
        f".env file not found: {ENV_FILE}"
    )

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini",
)

if not API_KEY:
    raise RuntimeError(
        f"OPENAI_API_KEY could not be loaded from {ENV_FILE}"
    )


# ---------------------------------------------------------------------
# OpenAI function tools
# ---------------------------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "name": "predict_readmission",
        "description": (
            "Run the local diabetes clinical prediction model to estimate "
            "30-day readmission probability. Use this when clinical risk "
            "information is needed and a clinical record is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "classify_activity",
        "description": (
            "Run the local PAMAP2 LSTM activity-recognition model on the "
            "available wearable sensor window. Use this when wearable "
            "activity information is needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@dataclass
class AgentResponse:
    """Structured response returned by the OpenAI agent."""

    status: str
    task: str | None
    final_text: str | None
    tool_results: dict[str, Any]
    decision_log: list[str]
    openai_metadata: dict[str, Any] | None
    safety_message: str | None = None


class OpenAIHealthDecisionSupportAgent:
    """Bounded OpenAI agent that can invoke local ML/DL tools."""

    def __init__(
        self,
        *,
        readmission_model_path: str | Path,
        activity_model_path: str | Path,
        model: str = MODEL_NAME,
        readmission_threshold: float = 0.50,
        max_tool_rounds: int = 6,
        client: OpenAI | None = None,
    ) -> None:

        if not 0.0 <= readmission_threshold <= 1.0:
            raise ValueError(
                "readmission_threshold must be between 0 and 1."
            )

        if max_tool_rounds <= 0:
            raise ValueError(
                "max_tool_rounds must be greater than zero."
            )

        self.readmission_model_path = Path(
            readmission_model_path
        )

        self.activity_model_path = Path(
            activity_model_path
        )

        self.model = model
        self.readmission_threshold = float(
            readmission_threshold
        )
        self.max_tool_rounds = int(
            max_tool_rounds
        )

        self.client = client or OpenAI(
            api_key=API_KEY
        )

    @staticmethod
    def _function_calls(
        response: Any,
    ) -> list[Any]:
        """Return function-call items from a Responses API response."""

        return [
            item
            for item in response.output
            if getattr(
                item,
                "type",
                None,
            ) == "function_call"
        ]

    @staticmethod
    def _required_tools_for_task(
        task: str,
    ) -> set[str]:
        """Return the tools expected for each supported task.

        This is used for evaluation and to remind the OpenAI model when an
        integrated assessment is incomplete. OpenAI still chooses the actual
        function calls through the Responses API.
        """

        if task == "clinical_risk":
            return {
                "predict_readmission"
            }

        if task == "activity_analysis":
            return {
                "classify_activity"
            }

        if task == "integrated_assessment":
            return {
                "predict_readmission",
                "classify_activity",
            }

        return set()

    def _initial_response(
        self,
        request: dict[str, Any],
    ) -> Any:
        """Send the task to OpenAI with the available local tools."""

        prompt = build_agent_input(
            task=request["task"],
            message=request.get(
                "message"
            ),
            has_clinical_record=(
                request.get(
                    "clinical_record"
                )
                is not None
            ),
            has_sensor_window=(
                request.get(
                    "sensor_window"
                )
                is not None
            ),
        )

        # Add explicit task-level expectation while still letting the model
        # choose the concrete function call(s).
        required_tools = sorted(
            self._required_tools_for_task(
                request["task"]
            )
        )

        if required_tools:
            prompt += (
                "\n\nFor this task, complete the analysis using the "
                "appropriate available local predictive tool(s). "
                f"Expected capabilities: {required_tools}. "
                "Do not provide a final assessment before obtaining the "
                "needed tool result(s)."
            )

        return self.client.responses.create(
            model=self.model,
            instructions=AGENT_INSTRUCTIONS,
            input=prompt,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )

    def _continue_after_tools(
        self,
        *,
        previous_response_id: str,
        tool_outputs: list[dict[str, Any]],
        missing_expected_tools: set[str],
    ) -> Any:
        """Return local tool outputs to OpenAI and continue reasoning."""

        continuation_input: list[dict[str, Any]] = list(
            tool_outputs
        )

        if missing_expected_tools:
            continuation_input.append(
                {
                    "role": "user",
                    "content": (
                        "Continue the requested assessment. "
                        "Before producing the final answer, use the remaining "
                        "relevant local capability or capabilities: "
                        f"{sorted(missing_expected_tools)}."
                    ),
                }
            )

        # Instructions are repeated because previous_response_id does not
        # automatically carry the prior `instructions` parameter forward.
        return self.client.responses.create(
            model=self.model,
            instructions=AGENT_INSTRUCTIONS,
            previous_response_id=previous_response_id,
            input=continuation_input,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )

    def run(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate, invoke OpenAI, execute local tools, and return final text."""

        decision_log: list[str] = []
        tool_results: dict[str, Any] = {}

        try:
            decision_log.append(
                "Received request."
            )

            safety = validate_request(
                request
            )

            decision_log.append(
                "Deterministic input validation completed."
            )

            if not safety.allowed:
                decision_log.append(
                    "Safety screening blocked OpenAI and local tool execution."
                )

                return asdict(
                    AgentResponse(
                        status="safety_fallback",
                        task=request.get(
                            "task"
                        ),
                        final_text=None,
                        tool_results={},
                        decision_log=decision_log,
                        openai_metadata=None,
                        safety_message=safety.reason,
                    )
                )

            decision_log.append(
                "Safety screening passed."
            )

            response = self._initial_response(
                request
            )

            decision_log.append(
                f"OpenAI response created: {response.id}."
            )

            expected_tools = (
                self._required_tools_for_task(
                    request["task"]
                )
            )

            for round_number in range(
                1,
                self.max_tool_rounds + 1,
            ):
                function_calls = (
                    self._function_calls(
                        response
                    )
                )

                # If OpenAI returned text before the expected tool(s), prompt
                # it once more instead of silently accepting a non-agentic
                # response.
                if not function_calls:
                    missing_expected = (
                        expected_tools
                        - set(
                            tool_results
                        )
                    )

                    if missing_expected:
                        decision_log.append(
                            "OpenAI returned text before completing expected "
                            f"tool use: {sorted(missing_expected)}."
                        )

                        response = (
                            self.client.responses.create(
                                model=self.model,
                                instructions=AGENT_INSTRUCTIONS,
                                previous_response_id=response.id,
                                input=(
                                    "The assessment is incomplete because the "
                                    "required local predictive result(s) have "
                                    "not yet been obtained. Use the available "
                                    "function tool(s) now before answering. "
                                    f"Remaining: {sorted(missing_expected)}."
                                ),
                                tools=OPENAI_TOOLS,
                                tool_choice="auto",
                            )
                        )

                        continue

                    final_text = (
                        response.output_text
                        or ""
                    ).strip()

                    if not final_text:
                        raise RuntimeError(
                            "OpenAI returned neither tool calls nor final text."
                        )

                    decision_log.append(
                        "OpenAI returned final decision-support explanation."
                    )

                    metadata = (
                        build_final_metadata(
                            model=self.model,
                            response_id=response.id,
                            tool_results=tool_results,
                        )
                    )

                    return asdict(
                        AgentResponse(
                            status="success",
                            task=request["task"],
                            final_text=final_text,
                            tool_results=tool_results,
                            decision_log=decision_log,
                            openai_metadata=metadata,
                            safety_message=None,
                        )
                    )

                decision_log.append(
                    "OpenAI selected "
                    f"{len(function_calls)} tool call(s) "
                    f"in round {round_number}."
                )

                tool_outputs: list[
                    dict[str, Any]
                ] = []

                for call in function_calls:
                    tool_name = call.name

                    decision_log.append(
                        f"OpenAI selected tool: {tool_name}."
                    )

                    # Validate tool argument JSON even though these tools are
                    # intentionally parameterless. Raw payloads stay local.
                    if call.arguments:
                        try:
                            parsed_arguments = (
                                json.loads(
                                    call.arguments
                                )
                            )
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON arguments for {tool_name}."
                            ) from exc

                        if parsed_arguments not in (
                            {},
                            None,
                        ):
                            decision_log.append(
                                f"Ignored unexpected arguments for {tool_name}; "
                                "raw inputs remain local."
                            )

                    # Avoid rerunning the same expensive local model if the
                    # model calls an already-completed function again.
                    if tool_name in tool_results:
                        result = tool_results[
                            tool_name
                        ]

                        decision_log.append(
                            f"Reused prior local result for {tool_name}."
                        )

                    else:
                        result = (
                            execute_agent_tool(
                                tool_name,
                                clinical_record=request.get(
                                    "clinical_record"
                                ),
                                sensor_window=request.get(
                                    "sensor_window"
                                ),
                                readmission_model_path=(
                                    self.readmission_model_path
                                ),
                                activity_model_path=(
                                    self.activity_model_path
                                ),
                                readmission_threshold=(
                                    self.readmission_threshold
                                ),
                            )
                        )

                        tool_results[
                            tool_name
                        ] = result

                        decision_log.append(
                            f"Local tool completed: {tool_name}."
                        )

                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(
                                result
                            ),
                        }
                    )

                missing_expected = (
                    expected_tools
                    - set(
                        tool_results
                    )
                )

                response = (
                    self._continue_after_tools(
                        previous_response_id=response.id,
                        tool_outputs=tool_outputs,
                        missing_expected_tools=missing_expected,
                    )
                )

                decision_log.append(
                    "Local tool output returned to OpenAI."
                )

            raise RuntimeError(
                "Maximum OpenAI tool-call rounds exceeded."
            )

        except (
            ValueError,
            TypeError,
            FileNotFoundError,
            RuntimeError,
        ) as exc:
            decision_log.append(
                "Request failed safely: "
                f"{type(exc).__name__}."
            )

            return asdict(
                AgentResponse(
                    status="invalid_input",
                    task=(
                        request.get(
                            "task"
                        )
                        if isinstance(
                            request,
                            dict,
                        )
                        else None
                    ),
                    final_text=None,
                    tool_results=tool_results,
                    decision_log=decision_log,
                    openai_metadata=None,
                    safety_message=str(
                        exc
                    ),
                )
            )


if __name__ == "__main__":
    print(
        f"Loaded environment file: {ENV_FILE}"
    )
    print(
        f"Using model: {MODEL_NAME}"
    )

    client = OpenAI(
        api_key=API_KEY
    )

    print(
        "OpenAI client initialized successfully."
    )
    print(
        "Function-calling agent is ready for integrated tests."
    )
