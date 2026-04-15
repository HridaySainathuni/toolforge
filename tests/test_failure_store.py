import pytest
import tempfile
import os
from agent.failure_store import FailureStore


@pytest.fixture
def tmp_failures():
    with tempfile.TemporaryDirectory() as d:
        yield FailureStore(db_path=os.path.join(d, "failures.db"))


def test_log_and_retrieve(tmp_failures):
    tmp_failures.log("sort a list", "def sort...", "NameError: sort not defined")
    recent = tmp_failures.get_recent("sort a list", limit=5)
    assert len(recent) == 1
    assert "NameError" in recent[0]["error_msg"]


def test_clear_removes_task_failures(tmp_failures):
    tmp_failures.log("sort a list", "def sort...", "error 1")
    tmp_failures.log("sort a list", "def sort2...", "error 2")
    tmp_failures.clear("sort a list")
    assert tmp_failures.get_recent("sort a list") == []


def test_get_recent_respects_limit(tmp_failures):
    for i in range(10):
        tmp_failures.log("task", f"code {i}", f"error {i}")
    recent = tmp_failures.get_recent("task", limit=3)
    assert len(recent) == 3
