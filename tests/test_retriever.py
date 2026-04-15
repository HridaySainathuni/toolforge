import pytest
import numpy as np
from agent.retriever import ToolRetriever, embed


def test_embed_returns_normalized_384d_vector():
    v = embed("find primes up to N")
    assert v.shape == (384,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_retrieve_returns_match_above_threshold(tmp_db, fake_embedding):
    # Store a tool using the embed of its description
    emb = embed("sort a list of integers in ascending order")
    tmp_db.add_tool(
        {
            "name": "sort_list",
            "description": "Sort a list of integers in ascending order",
            "source_code": "def sort_list(items: list) -> list:\n    return sorted(items)",
            "args": {"items": "list"},
            "returns": "list",
            "tags": ["sort"],
        },
        embedding=emb,
        task_context="test",
    )
    retriever = ToolRetriever(library=tmp_db, threshold=0.7)
    results = retriever.retrieve("sort a list of integers", top_k=3)
    assert results is not None
    assert results[0]["name"] == "sort_list"


def test_retrieve_returns_none_when_nothing_above_threshold(tmp_db, fake_embedding):
    emb = embed("calculate tax on invoices")
    tmp_db.add_tool(
        {
            "name": "calc_tax",
            "description": "Calculate tax on invoices",
            "source_code": "def calc_tax(amount: float) -> float:\n    return amount * 0.1",
            "args": {"amount": "float"},
            "returns": "float",
            "tags": ["tax"],
        },
        embedding=emb,
        task_context="test",
    )
    retriever = ToolRetriever(library=tmp_db, threshold=0.99)
    results = retriever.retrieve("sort a list of integers", top_k=3)
    assert results is None
