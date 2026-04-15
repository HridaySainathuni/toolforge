import json
import pytest
from unittest.mock import MagicMock, patch
from agent.tool_generator import ToolGenerator


VALID_SPEC = {
    "function_name": "add_numbers",
    "name": "add_numbers",
    "source_code": "def add_numbers(a: int, b: int) -> int:\n    \"\"\"Add two integers.\"\"\"\n    return a + b",
    "description": "Add two integers",
    "args": {"a": "int — first number", "b": "int — second number"},
    "returns": "int — the sum",
    "tags": ["math", "add"],
    "test_call": {"a": 1, "b": 2},
}


@pytest.fixture
def mock_generator():
    gen = ToolGenerator.__new__(ToolGenerator)
    gen.client = MagicMock()
    return gen


def _make_response(data: dict):
    """Helper: make a mock Anthropic API response containing JSON."""
    return MagicMock(content=[MagicMock(text=json.dumps(data))])


def test_generate_returns_spec_on_valid_llm_response(mock_generator):
    mock_generator.client.messages.create.return_value = _make_response(VALID_SPEC)
    result = mock_generator.generate(
        capability_needed="add numbers",
        capability_detail="add two integers a and b",
        task_context="compute 2+3",
    )
    assert result is not None
    assert result["name"] == "add_numbers"


def test_generate_returns_none_on_json_parse_failure(mock_generator):
    mock_generator.client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="this is not json at all")]
    )
    result = mock_generator.generate(
        capability_needed="add numbers",
        capability_detail="add two integers",
        task_context="compute 2+3",
    )
    assert result is None


def test_generate_includes_failures_in_prompt(mock_generator):
    mock_generator.client.messages.create.return_value = _make_response(VALID_SPEC)
    failures = [{"attempted_code": "def bad(): pass", "error_msg": "TypeError: missing args"}]
    mock_generator.generate(
        capability_needed="add numbers",
        capability_detail="add two integers",
        task_context="compute 2+3",
        failures=failures,
    )
    # Verify the user message contained the failure text
    call_args = mock_generator.client.messages.create.call_args
    messages = call_args[1]["messages"]
    user_content = messages[0]["content"]
    assert "TypeError: missing args" in user_content


def test_generate_with_no_failures_still_works(mock_generator):
    mock_generator.client.messages.create.return_value = _make_response(VALID_SPEC)
    result = mock_generator.generate(
        capability_needed="add numbers",
        capability_detail="add two integers",
        task_context="compute 2+3",
        failures=None,
    )
    assert result is not None


def test_generate_strips_markdown_fences(mock_generator):
    """LLM sometimes wraps JSON in code fences — should still parse."""
    fenced = "```json\n" + json.dumps(VALID_SPEC) + "\n```"
    mock_generator.client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=fenced)]
    )
    result = mock_generator.generate(
        capability_needed="add numbers",
        capability_detail="add two integers",
        task_context="compute 2+3",
    )
    assert result is not None
