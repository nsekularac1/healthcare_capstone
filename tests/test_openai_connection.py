from pathlib import Path
import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(
    PROJECT_ROOT / ".env",
    override=True,
)


@pytest.mark.integration
def test_openai_connection():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv(
        "OPENAI_MODEL",
        "gpt-4.1-mini",
    )

    if not api_key:
        pytest.skip(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key
    )

    response = client.responses.create(
        model=model,
        input="Reply with exactly: SUCCESS",
    )

    assert "SUCCESS" in response.output_text.upper()
