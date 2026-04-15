import pytest
import numpy as np
from library.tool_library import ToolLibrary

FAKE_TOOL = {
    "name": "add_numbers",
    "description": "Add two integers together",
    "source_code": "def add_numbers(a: int, b: int) -> int:\n    return a + b",
    "args": {"a": "int — first number", "b": "int — second number"},
    "returns": "int — the sum",
    "tags": ["math", "add", "arithmetic"],
    "test_call": {"a": 1, "b": 2},
}

FAKE_EMBEDDING = np.random.rand(384).astype(np.float32)
FAKE_EMBEDDING /= np.linalg.norm(FAKE_EMBEDDING)


def test_add_and_retrieve_by_name(tmp_db):
    tmp_db.add_tool(FAKE_TOOL, embedding=FAKE_EMBEDDING, task_context="test task")
    code = tmp_db.get_source_code("add_numbers")
    assert code == FAKE_TOOL["source_code"]


def test_increment_use(tmp_db):
    tmp_db.add_tool(FAKE_TOOL, embedding=FAKE_EMBEDDING, task_context="test")
    tmp_db.increment_use("add_numbers")
    tools = tmp_db.get_all_tools_public()
    assert tools[0]["use_count"] == 1


def test_search_returns_similar(tmp_db):
    tmp_db.add_tool(FAKE_TOOL, embedding=FAKE_EMBEDDING, task_context="test")
    results = tmp_db.search(query_embedding=FAKE_EMBEDDING, top_k=5, threshold=0.5)
    assert len(results) == 1
    assert results[0]["name"] == "add_numbers"


def test_search_below_threshold_returns_empty(tmp_db):
    tmp_db.add_tool(FAKE_TOOL, embedding=FAKE_EMBEDDING, task_context="test")
    opposite = np.ones(384, dtype=np.float32) * -1
    opposite /= np.linalg.norm(opposite)
    results = tmp_db.search(query_embedding=opposite, top_k=5, threshold=0.5)
    assert results == []


def test_replace_tool(tmp_db):
    tmp_db.add_tool(FAKE_TOOL, embedding=FAKE_EMBEDDING, task_context="test")
    new_code = "def add_numbers(a: int, b: int) -> int:\n    return b + a"
    tmp_db.replace_tool("add_numbers", new_source=new_code, new_embedding=FAKE_EMBEDDING)
    assert tmp_db.get_source_code("add_numbers") == new_code


def test_delete_tool(tmp_db):
    tmp_db.add_tool(FAKE_TOOL, embedding=FAKE_EMBEDDING, task_context="test")
    tmp_db.delete_tool("add_numbers")
    assert tmp_db.get_source_code("add_numbers") is None


def test_get_all_tools_for_prompt_excludes_source(tmp_db):
    tmp_db.add_tool(FAKE_TOOL, embedding=FAKE_EMBEDDING, task_context="test")
    prompt_tools = tmp_db.get_all_tools_for_prompt()
    assert len(prompt_tools) == 1
    assert "source_code" not in prompt_tools[0]
    assert "name" in prompt_tools[0]
    assert "description" in prompt_tools[0]
