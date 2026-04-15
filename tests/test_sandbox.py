import pytest
from agent.sandbox import run_in_sandbox


def test_basic_execution():
    result = run_in_sandbox(
        source_code="def add(a: int, b: int) -> int:\n    return a + b",
        function_name="add",
        args={"a": 2, "b": 3},
        timeout=5,
    )
    assert result.success
    assert result.result == "5"


def test_timeout():
    result = run_in_sandbox(
        source_code="def slow():\n    import time; time.sleep(100)",
        function_name="slow",
        args={},
        timeout=2,
    )
    assert not result.success
    assert "timed out" in result.error.lower()


def test_runtime_error_returns_failure():
    result = run_in_sandbox(
        source_code="def boom():\n    raise ValueError('oops')",
        function_name="boom",
        args={},
        timeout=5,
    )
    assert not result.success


def test_returns_string_for_list():
    result = run_in_sandbox(
        source_code="def get_list() -> list:\n    return [1, 2, 3]",
        function_name="get_list",
        args={},
        timeout=5,
    )
    assert result.success
    assert result.result == "[1, 2, 3]"


def test_missing_function_name():
    result = run_in_sandbox(
        source_code="def other_func():\n    return 42",
        function_name="nonexistent_func",
        args={},
        timeout=5,
    )
    assert not result.success
