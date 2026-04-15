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


def test_add_and_retrieve_by_name(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test task")
    code = tmp_db.get_source_code("add_numbers")
    assert code == FAKE_TOOL["source_code"]


def test_increment_use(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test")
    tmp_db.increment_use("add_numbers")
    tools = tmp_db.get_all_tools_public()
    assert tools[0]["use_count"] == 1


def test_search_returns_similar(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test")
    results = tmp_db.search(query_embedding=fake_embedding, top_k=5, threshold=0.5)
    assert len(results) == 1
    assert results[0]["name"] == "add_numbers"


def test_search_below_threshold_returns_empty(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test")
    # Build an orthogonal vector (dot product = 0, well below any positive threshold)
    perp = np.zeros(384, dtype=np.float32)
    perp[0] = -fake_embedding[1]
    perp[1] = fake_embedding[0]
    norm = np.linalg.norm(perp)
    if norm > 0:
        perp = perp / norm
    results = tmp_db.search(query_embedding=perp, top_k=5, threshold=0.5)
    assert results == []


def test_replace_tool(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test")
    new_code = "def add_numbers(a: int, b: int) -> int:\n    return b + a"
    tmp_db.replace_tool("add_numbers", new_source=new_code, new_embedding=fake_embedding)
    assert tmp_db.get_source_code("add_numbers") == new_code


def test_delete_tool(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test")
    tmp_db.delete_tool("add_numbers")
    assert tmp_db.get_source_code("add_numbers") is None


def test_get_all_tools_for_prompt_excludes_source(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test")
    prompt_tools = tmp_db.get_all_tools_for_prompt()
    assert len(prompt_tools) == 1
    assert "source_code" not in prompt_tools[0]
    assert "name" in prompt_tools[0]
    assert "description" in prompt_tools[0]


def test_record_outcome_increments_count(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test")
    tmp_db.record_outcome("add_numbers", success=True)
    tmp_db.record_outcome("add_numbers", success=False)
    tools = tmp_db.get_all_tools_public()
    assert tools[0]["use_count"] == 2


def test_get_all_returns_all_tools(tmp_db, fake_embedding):
    tmp_db.add_tool(FAKE_TOOL, embedding=fake_embedding, task_context="test")
    all_tools = tmp_db.get_all()
    assert len(all_tools) == 1
    assert all_tools[0]["name"] == "add_numbers"
    assert "source_code" in all_tools[0]
