# ToolForge: Complete the Research System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade ToolForge from keyword-based retrieval to semantic retrieval, add failure memory, build the librarian agent, and implement the full evaluation harness needed to test H1–H4 research hypotheses.

**Architecture:** The existing Flask+AgentLoop forms the working core. We layer semantic embeddings (sentence-transformers) onto a new SQLite-backed ToolLibrary, add a FailureStore for retry context, build a Librarian agent that clusters and merges redundant tools, and construct an evaluation harness that runs 4 ablation conditions across task batches.

**Tech Stack:** Python 3.11+, Anthropic Claude API (claude-sonnet-4-6), sentence-transformers (all-MiniLM-L6-v2, 384-dim), SQLite3, pytest, pandas + matplotlib + seaborn for analysis, Flask (existing).

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `library/tool_library.py` | **Rewrite** | SQLite backend, embedding storage, semantic search |
| `requirements.txt` | **Modify** | Add sentence-transformers, pytest, seaborn |
| `config.py` | **Modify** | Add RETRIEVAL_THRESHOLD, LIBRARIAN_THRESHOLD, ablation flags |
| `agent/retriever.py` | **New** | Embed query, cosine search, threshold gating |
| `agent/failure_store.py` | **New** | SQLite log of failed tool attempts |
| `agent/librarian.py` | **New** | Cluster + merge redundant tools via LLM |
| `agent/prompts.py` | **Modify** | Add LIBRARIAN_SYSTEM_PROMPT, inject failures into tool-gen prompt |
| `agent/loop.py` | **Modify** | Wire in retriever + failure store |
| `agent/tool_generator.py` | **Modify** | Accept `failures` list, pass to prompt builder |
| `web/app.py` | **Modify** | Add `/api/librarian/run` endpoint |
| `eval/__init__.py` | **New** | Package marker |
| `eval/tasks/math_batch1.jsonl` | **New** | 50 math tasks with ground truth |
| `eval/tasks/text_batch2.jsonl` | **New** | 50 text-processing tasks |
| `eval/tasks/mixed_batch3.jsonl` | **New** | 50 cross-domain tasks (tests generalization) |
| `eval/run_eval.py` | **New** | Runs agent over task batches, logs CSV results |
| `eval/analyze.py` | **New** | Loads CSV, computes 6 metrics, produces plots |
| `tests/conftest.py` | **New** | Shared pytest fixtures (tmp db, mock LLM) |
| `tests/test_tool_library.py` | **New** | Unit tests for SQLite library |
| `tests/test_retriever.py` | **New** | Unit tests for semantic retrieval |
| `tests/test_failure_store.py` | **New** | Unit tests for failure logging |
| `tests/test_sandbox.py` | **New** | Unit tests for sandbox execution |

---

## Task 1: Migrate ToolLibrary to SQLite + Semantic Embeddings

**This is the foundation. Everything else depends on it.**

**Files:**
- Rewrite: `library/tool_library.py`
- Modify: `requirements.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_tool_library.py`

- [ ] **Step 1: Add dependencies to requirements.txt**

```
anthropic>=0.25.0
flask>=3.0.0
python-dotenv>=1.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
pandas>=2.0.0
numpy>=1.25.0
sentence-transformers>=2.7.0
pytest>=8.0.0
seaborn>=0.13.0
matplotlib>=3.8.0
```

Install: `pip install -r requirements.txt`

- [ ] **Step 2: Write the failing tests in `tests/test_tool_library.py`**

```python
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
    # Search with the exact same embedding — must score 1.0
    results = tmp_db.search(query_embedding=FAKE_EMBEDDING, top_k=5, threshold=0.5)
    assert len(results) == 1
    assert results[0]["name"] == "add_numbers"


def test_search_below_threshold_returns_empty(tmp_db):
    tmp_db.add_tool(FAKE_TOOL, embedding=FAKE_EMBEDDING, task_context="test")
    opposite = -FAKE_EMBEDDING  # cosine sim = -1
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
```

- [ ] **Step 3: Write the `tests/conftest.py` with shared fixtures**

```python
import pytest
import tempfile
import os
import numpy as np
from library.tool_library import ToolLibrary


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_tools.db")
        lib = ToolLibrary(db_path=db_path, seed=False)
        yield lib
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /Users/prithviraj/Documents/CS/UVA/toolforge
pytest tests/test_tool_library.py -v
```

Expected: `ImportError` or `AttributeError` — the new API doesn't exist yet.

- [ ] **Step 5: Rewrite `library/tool_library.py`**

```python
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np


class ToolLibrary:
    def __init__(self, db_path: str = "library/tool_library.db", seed: bool = True) -> None:
        self.db_path = db_path
        self._init_db()
        if seed and self._is_empty():
            self._seed()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tools (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    name         TEXT NOT NULL UNIQUE,
                    description  TEXT NOT NULL,
                    source_code  TEXT NOT NULL,
                    embedding    BLOB NOT NULL,
                    args         TEXT NOT NULL DEFAULT '{}',
                    returns      TEXT NOT NULL DEFAULT '',
                    tags         TEXT NOT NULL DEFAULT '[]',
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    origin_task  TEXT DEFAULT '',
                    reuse_count  INTEGER DEFAULT 0,
                    pass_rate    REAL DEFAULT 1.0
                );
            """)

    def _is_empty(self) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM tools").fetchone()
            return row[0] == 0

    # --- Write operations ---

    def add_tool(
        self,
        tool_spec: dict[str, Any],
        embedding: np.ndarray,
        task_context: str = "",
    ) -> int:
        """Insert or replace a tool. Returns the row id."""
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO tools
                   (name, description, source_code, embedding, args, returns, tags, origin_task)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tool_spec["name"],
                    tool_spec.get("description", ""),
                    tool_spec["source_code"],
                    embedding.astype(np.float32).tobytes(),
                    json.dumps(tool_spec.get("args", {})),
                    tool_spec.get("returns", ""),
                    json.dumps(tool_spec.get("tags", [])),
                    task_context,
                ),
            )
            return cursor.lastrowid

    def replace_tool(self, name: str, new_source: str, new_embedding: np.ndarray) -> None:
        """Replace source code and embedding of an existing tool."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE tools SET source_code=?, embedding=? WHERE name=?",
                (new_source, new_embedding.astype(np.float32).tobytes(), name),
            )

    def delete_tool(self, name: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM tools WHERE name=?", (name,))

    def increment_use(self, name: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE tools SET reuse_count = reuse_count + 1 WHERE name=?", (name,)
            )

    def update_pass_rate(self, name: str, success: bool) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT reuse_count, pass_rate FROM tools WHERE name=?", (name,)
            ).fetchone()
            if row:
                n = row["reuse_count"] or 1
                new_rate = (row["pass_rate"] * n + (1.0 if success else 0.0)) / (n + 1)
                conn.execute(
                    "UPDATE tools SET pass_rate=? WHERE name=?", (new_rate, name)
                )

    # --- Read operations ---

    def get_source_code(self, name: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT source_code FROM tools WHERE name=?", (name,)
            ).fetchone()
            return row["source_code"] if row else None

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> list[dict[str, Any]]:
        """Return top-k tools above cosine similarity threshold."""
        q = query_embedding.astype(np.float32)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, description, source_code, embedding, args, returns, tags FROM tools"
            ).fetchall()

        scored = []
        for row in rows:
            stored = np.frombuffer(row["embedding"], dtype=np.float32)
            sim = float(np.dot(q, stored))
            if sim >= threshold:
                scored.append((sim, self._row_to_dict(row)))

        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:top_k]]

    def get_all_tools_for_prompt(self) -> list[dict[str, Any]]:
        """Return compact tool list for injection into agent system prompt."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, description, args, returns FROM tools"
            ).fetchall()
        return [
            {
                "name": r["name"],
                "description": r["description"],
                "args": json.loads(r["args"]),
                "returns": r["returns"],
            }
            for r in rows
        ]

    def get_all_tools_public(self) -> list[dict[str, Any]]:
        """Return all tools for the web API (no source code)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT name, description, args, returns, tags, reuse_count, created_at, origin_task
                   FROM tools ORDER BY reuse_count DESC"""
            ).fetchall()
        return [
            {
                "name": r["name"],
                "description": r["description"],
                "args": json.loads(r["args"]),
                "returns": r["returns"],
                "tags": json.loads(r["tags"]),
                "use_count": r["reuse_count"],
                "created_at": r["created_at"],
                "created_for_task": r["origin_task"],
            }
            for r in rows
        ]

    def get_all(self) -> list[dict[str, Any]]:
        """Return all tools including source code (for librarian)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM tools").fetchall()
        return [dict(r) for r in rows]

    # --- Private helpers ---

    def _row_to_dict(self, row) -> dict[str, Any]:
        return {
            "name": row["name"],
            "description": row["description"],
            "source_code": row["source_code"],
            "args": json.loads(row["args"]),
            "returns": row["returns"],
            "tags": json.loads(row["tags"]),
        }

    def _seed(self) -> None:
        """Seed the library with 5 starter tools (requires embedder)."""
        # Import here to avoid circular dependency — embedder lives in agent/retriever.py
        # Seeding with a zero embedding; retriever will re-embed on first search.
        zero_emb = np.zeros(384, dtype=np.float32)
        seeds = [
            {
                "name": "read_file",
                "description": "Read text content from a local file",
                "source_code": (
                    "def read_file(path: str) -> str:\n"
                    '    """Read and return the text content of a file."""\n'
                    "    try:\n"
                    "        with open(path, 'r', encoding='utf-8') as f:\n"
                    "            return f.read()\n"
                    "    except Exception as e:\n"
                    "        return f'Error reading file: {e}'\n"
                ),
                "args": {"path": "str — path to the file"},
                "returns": "str — file content or error message",
                "tags": ["file", "read", "text", "local"],
            },
            {
                "name": "write_file",
                "description": "Write text content to a local file",
                "source_code": (
                    "def write_file(path: str, content: str) -> str:\n"
                    '    """Write content to a file, creating directories if needed."""\n'
                    "    import os\n"
                    "    try:\n"
                    "        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
                    "        with open(path, 'w', encoding='utf-8') as f:\n"
                    "            f.write(content)\n"
                    "        return f'Successfully wrote {len(content)} chars to {path}'\n"
                    "    except Exception as e:\n"
                    "        return f'Error writing file: {e}'\n"
                ),
                "args": {"path": "str — path to write to", "content": "str — text to write"},
                "returns": "str — success or error message",
                "tags": ["file", "write", "text", "local", "save"],
            },
            {
                "name": "fetch_webpage",
                "description": "Fetch the raw HTML content of a URL",
                "source_code": (
                    "def fetch_webpage(url: str) -> str:\n"
                    '    """Fetch and return the HTML content of a webpage."""\n'
                    "    import requests\n"
                    "    try:\n"
                    "        resp = requests.get(url, timeout=10, headers={'User-Agent': 'ToolForge/1.0'})\n"
                    "        resp.raise_for_status()\n"
                    "        return resp.text[:50000]\n"
                    "    except Exception as e:\n"
                    "        return f'Error fetching URL: {e}'\n"
                ),
                "args": {"url": "str — the URL to fetch"},
                "returns": "str — HTML content (truncated to 50k chars) or error message",
                "tags": ["web", "http", "fetch", "html", "url"],
            },
            {
                "name": "run_python_expression",
                "description": "Evaluate a Python expression and return the result",
                "source_code": (
                    "def run_python_expression(expression: str) -> str:\n"
                    '    """Safely evaluate a Python expression."""\n'
                    "    import math\n"
                    "    allowed = {'__builtins__': {}, 'math': math, 'abs': abs, 'round': round,\n"
                    "               'min': min, 'max': max, 'sum': sum, 'len': len,\n"
                    "               'int': int, 'float': float, 'str': str, 'list': list,\n"
                    "               'sorted': sorted, 'range': range, 'True': True,\n"
                    "               'False': False, 'None': None}\n"
                    "    try:\n"
                    "        return str(eval(expression, allowed))\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {"expression": "str — Python expression to evaluate"},
                "returns": "str — result or error message",
                "tags": ["math", "calculate", "eval", "expression"],
            },
            {
                "name": "search_text",
                "description": "Search text using a regex pattern and return all matches",
                "source_code": (
                    "def search_text(text: str, pattern: str) -> str:\n"
                    '    """Search text with a regex pattern, return JSON list."""\n'
                    "    import re, json\n"
                    "    try:\n"
                    "        return json.dumps(re.findall(pattern, text)[:100])\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {"text": "str — text to search", "pattern": "str — regex pattern"},
                "returns": "str — JSON array of matches",
                "tags": ["regex", "search", "text", "pattern", "find"],
            },
        ]
        for spec in seeds:
            self.add_tool(spec, embedding=zero_emb, task_context="seed")
```

- [ ] **Step 6: Run the tests and verify they pass**

```bash
pytest tests/test_tool_library.py -v
```

Expected output: all 7 tests PASS.

- [ ] **Step 7: Update config.py to use `.db` extension**

In `config.py`, change:
```python
LIBRARY_PATH: str = os.path.join(os.path.dirname(__file__), "library", "tool_library.db")
```

- [ ] **Step 8: Commit**

```bash
git add library/tool_library.py requirements.txt config.py tests/conftest.py tests/test_tool_library.py
git commit -m "Migrate ToolLibrary from JSON to SQLite with embedding storage"
```

---

## Task 2: Add Semantic ToolRetriever

**Files:**
- Create: `agent/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests in `tests/test_retriever.py`**

```python
import pytest
import numpy as np
from unittest.mock import MagicMock
from agent.retriever import ToolRetriever, embed


def test_embed_returns_normalized_384d_vector():
    v = embed("find primes up to N")
    assert v.shape == (384,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_retrieve_returns_match_above_threshold(tmp_db):
    # Add a tool whose description embeds close to our query
    query = "sort a list of integers"
    emb = embed("sort a list of integers in ascending order")
    tmp_db.add_tool(
        {
            "name": "sort_list",
            "description": "Sort a list of integers in ascending order",
            "source_code": "def sort_list(items): return sorted(items)",
            "args": {"items": "list"},
            "returns": "list",
            "tags": ["sort"],
        },
        embedding=emb,
        task_context="test",
    )
    retriever = ToolRetriever(library=tmp_db, threshold=0.7)
    results = retriever.retrieve(query, top_k=3)
    assert results is not None
    assert results[0]["name"] == "sort_list"


def test_retrieve_returns_none_when_nothing_above_threshold(tmp_db):
    emb = embed("calculate tax on invoices")
    tmp_db.add_tool(
        {
            "name": "calc_tax",
            "description": "Calculate tax on invoices",
            "source_code": "def calc_tax(amount): return amount * 0.1",
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.retriever'`

- [ ] **Step 3: Create `agent/retriever.py`**

```python
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from library.tool_library import ToolLibrary

log = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the embedding model (384-dim, ~80 MB)."""
    log.info("Loading sentence-transformers model: %s", _MODEL_NAME)
    return SentenceTransformer(_MODEL_NAME)


def embed(text: str) -> np.ndarray:
    """Embed a string as a normalized float32 384-dim vector."""
    model = _load_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


class ToolRetriever:
    def __init__(self, library: ToolLibrary, threshold: float = 0.75) -> None:
        self.library = library
        self.threshold = threshold

    def retrieve(self, task: str, top_k: int = 3) -> list[dict[str, Any]] | None:
        """Return top-k tools above threshold, or None if nothing qualifies.

        Returns None to signal the orchestrator to create a new tool.
        """
        query_emb = embed(task)
        results = self.library.search(
            query_embedding=query_emb,
            top_k=top_k,
            threshold=self.threshold,
        )
        if not results:
            log.debug("Retriever: no tools above threshold %.2f for '%s'", self.threshold, task[:60])
            return None
        log.debug("Retriever: found %d tools for '%s'", len(results), task[:60])
        return results
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_retriever.py -v
```

Note: first run downloads ~80 MB model. Subsequent runs are instant (cached).

Expected: all 3 tests PASS.

- [ ] **Step 5: Re-seed the library with real embeddings**

The seeded tools were stored with zero embeddings. Fix them:

```python
# Run once to re-embed seed tools
# python scripts/reseed_embeddings.py
```

Create `scripts/reseed_embeddings.py`:

```python
"""One-time script to re-embed the seed tools in the library."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from library.tool_library import ToolLibrary
from agent.retriever import embed

lib = ToolLibrary(db_path=Config.LIBRARY_PATH, seed=True)
for tool in lib.get_all():
    desc = tool["description"]
    emb = embed(desc)
    lib.replace_tool(tool["name"], new_source=tool["source_code"], new_embedding=emb)
    print(f"Re-embedded: {tool['name']}")
print("Done.")
```

```bash
python scripts/reseed_embeddings.py
```

- [ ] **Step 6: Commit**

```bash
git add agent/retriever.py tests/test_retriever.py scripts/reseed_embeddings.py
git commit -m "Add semantic ToolRetriever with sentence-transformers embeddings"
```

---

## Task 3: Add FailureStore

**Files:**
- Create: `agent/failure_store.py`
- Modify: `agent/prompts.py` (inject failures into tool-gen prompt)
- Modify: `agent/tool_generator.py` (accept `failures` list)
- Modify: `agent/loop.py` (log failures, pass to generator)
- Create: `tests/test_failure_store.py`

- [ ] **Step 1: Write failing tests in `tests/test_failure_store.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_failure_store.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `agent/failure_store.py`**

```python
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any
import os


class FailureStore:
    """Append-only log of failed tool-creation attempts.

    Injected into future tool-creation prompts so the agent avoids
    repeating the same mistake.
    """

    def __init__(self, db_path: str = "library/failures.db") -> None:
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    task         TEXT NOT NULL,
                    attempted_code TEXT,
                    error_msg    TEXT,
                    timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def log(self, task: str, attempted_code: str, error_msg: str) -> None:
        """Record a failed tool-creation attempt."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO failures (task, attempted_code, error_msg) VALUES (?, ?, ?)",
                (task, attempted_code, error_msg),
            )

    def get_recent(self, task: str, limit: int = 3) -> list[dict[str, Any]]:
        """Return the most recent failures for a given task."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT attempted_code, error_msg FROM failures
                   WHERE task=? ORDER BY timestamp DESC LIMIT ?""",
                (task, limit),
            ).fetchall()
        return [{"attempted_code": r["attempted_code"], "error_msg": r["error_msg"]} for r in rows]

    def clear(self, task: str) -> None:
        """Remove all failure records for a task (called on success)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM failures WHERE task=?", (task,))
```

- [ ] **Step 4: Update `agent/prompts.py` — add failures injection**

Add this function at the bottom of `prompts.py`:

```python
def build_tool_gen_user_prompt_with_failures(
    capability_needed: str,
    capability_detail: str,
    task_context: str,
    failures: list[dict],
) -> tuple[str, str]:
    """Like build_tool_gen_user_prompt but injects past failures."""
    name, _ = build_tool_gen_user_prompt(
        capability_needed, capability_detail, task_context
    )
    # name is returned as second element; rebuild with failures appended
    function_name = capability_needed.lower().replace(" ", "_").replace("-", "_")
    function_name = "".join(c for c in function_name if c.isalnum() or c == "_")

    failure_block = ""
    if failures:
        parts = []
        for i, f in enumerate(failures, 1):
            parts.append(
                f"PREVIOUS ATTEMPT {i}:\nCode:\n{f['attempted_code']}\nError: {f['error_msg']}"
            )
        failure_block = "\n\nPREVIOUS FAILED ATTEMPTS (avoid repeating these mistakes):\n" + "\n\n".join(parts)

    prompt = TOOL_GENERATOR_USER_PROMPT.format(
        capability_needed=capability_needed,
        capability_detail=capability_detail,
        task_context=task_context,
        function_name=function_name,
    ) + failure_block

    return prompt, function_name
```

- [ ] **Step 5: Update `agent/tool_generator.py` — accept failures list**

Change the `generate` method signature and prompt building:

```python
def generate(
    self,
    capability_needed: str,
    capability_detail: str,
    task_context: str,
    failures: list[dict] | None = None,
) -> dict[str, Any] | None:
    from agent.prompts import build_tool_gen_user_prompt_with_failures
    user_prompt, function_name = build_tool_gen_user_prompt_with_failures(
        capability_needed, capability_detail, task_context, failures or []
    )
    # ... rest unchanged
```

- [ ] **Step 6: Update `agent/loop.py` — wire in failure store**

In `AgentLoop.__init__`, add:

```python
from agent.failure_store import FailureStore
# ...
self.failure_store = FailureStore()
```

In `_handle_acquire_tool`, before calling `self.generator.generate(...)`, fetch failures:

```python
failures = self.failure_store.get_recent(task, limit=3)

spec = self.generator.generate(
    capability_needed=capability_needed,
    capability_detail=capability_detail,
    task_context=task,
    failures=failures,
)
```

If spec is None (all attempts failed), log the failure:

```python
if spec is None:
    self.failure_store.log(task, attempted_code="<generation failed>", error_msg="LLM refused or produced invalid Python")
    self._emit("tool_acquisition_failed", {"capability": capability_needed})
    return None
```

After successful `add_tool`, clear failures:

```python
self.library.add_tool(spec, embedding=embed(spec["description"]), task_context=task)
self.failure_store.clear(task)
```

Note: import `embed` from `agent.retriever` at top of `loop.py`.

- [ ] **Step 7: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass (test_tool_library, test_retriever, test_failure_store).

- [ ] **Step 8: Commit**

```bash
git add agent/failure_store.py agent/prompts.py agent/tool_generator.py agent/loop.py tests/test_failure_store.py
git commit -m "Add FailureStore and wire failure injection into tool generation prompts"
```

---

## Task 4: Wire Semantic Retrieval into AgentLoop

**The agent currently passes all tools to Claude for it to decide. We need semantic pre-filtering.**

**Files:**
- Modify: `agent/loop.py` (add retriever, use it in `_call_claude`)
- Modify: `config.py` (add RETRIEVAL_THRESHOLD, ABLATION_NO_LIBRARY flag)

- [ ] **Step 1: Update `config.py`**

```python
class Config:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MODEL: str = "claude-sonnet-4-6"
    MAX_ITERATIONS: int = 20
    SANDBOX_TIMEOUT: int = 10
    VALIDATION_TIMEOUT: int = 15
    TOOL_GEN_RETRIES: int = 3
    LIBRARY_PATH: str = os.path.join(os.path.dirname(__file__), "library", "tool_library.db")
    FAILURES_PATH: str = os.path.join(os.path.dirname(__file__), "library", "failures.db")
    PORT: int = 5000
    # Retrieval
    RETRIEVAL_THRESHOLD: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.75"))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    # Ablation flags
    ABLATION_NO_LIBRARY: bool = os.getenv("ABLATION_NO_LIBRARY", "false").lower() == "true"
    ABLATION_NO_ABSTRACTION: bool = os.getenv("ABLATION_NO_ABSTRACTION", "false").lower() == "true"
    ABLATION_NO_LIBRARIAN: bool = os.getenv("ABLATION_NO_LIBRARIAN", "false").lower() == "true"
```

- [ ] **Step 2: Update `agent/loop.py` — add retriever, use semantic pre-filtering**

In `AgentLoop.__init__`, add retriever:

```python
from agent.retriever import ToolRetriever, embed
from agent.failure_store import FailureStore
# ...
self.retriever = ToolRetriever(library=tool_library, threshold=Config.RETRIEVAL_THRESHOLD)
self.failure_store = FailureStore(db_path=Config.FAILURES_PATH)
```

In `_call_claude`, replace `get_all_tools_for_prompt()` with semantic pre-filtering:

```python
def _call_claude(self, task: str) -> dict[str, Any] | None:
    if Config.ABLATION_NO_LIBRARY:
        tools_for_prompt = []
    else:
        # Semantically retrieve relevant tools rather than passing all tools
        relevant = self.retriever.retrieve(task, top_k=Config.RETRIEVAL_TOP_K)
        if relevant:
            tools_for_prompt = [
                {"name": t["name"], "description": t["description"],
                 "args": t["args"], "returns": t["returns"]}
                for t in relevant
            ]
        else:
            tools_for_prompt = []
    # ... rest of _call_claude unchanged
```

- [ ] **Step 3: Manual smoke test — run the agent on 3 tasks**

```bash
# Start the server
python main.py &

# Test via curl (or use the web UI at http://localhost:5000)
curl -X POST http://localhost:5000/api/task \
  -H "Content-Type: application/json" \
  -d '{"task": "What is 17 * 23?"}'
```

Check logs to confirm: `Retriever: found N tools for 'What is 17 * 23?'`

- [ ] **Step 4: Commit**

```bash
git add config.py agent/loop.py
git commit -m "Wire semantic retrieval into agent loop; add ablation config flags"
```

---

## Task 5: Add Librarian Agent

**Files:**
- Create: `agent/librarian.py`
- Modify: `agent/prompts.py` (add LIBRARIAN_SYSTEM_PROMPT)
- Modify: `web/app.py` (add `/api/librarian/run` endpoint)

- [ ] **Step 1: Add librarian prompt to `agent/prompts.py`**

Add to the top of `prompts.py` (as a constant):

```python
LIBRARIAN_SYSTEM_PROMPT = """You are a code librarian. You will receive a list of Python tools from a tool library.

Your job:
1. Identify tools that are REDUNDANT or HIGHLY OVERLAPPING in functionality.
2. For each redundant pair/group, propose a single merged function that replaces both, is MORE GENERAL than either, and has a clear docstring.
3. Identify tools that are too SPECIFIC to one task and suggest a more abstract version.

Rules for merged/refactored tools:
- ALL imports must be inside the function body
- The function must have type annotations and a docstring
- The new function must correctly handle all use cases of the tools it replaces
- Keep the most informative name (not both names)

Output ONLY a JSON object with no markdown:
{
  "merges": [
    {
      "replace_names": ["name_a", "name_b"],
      "new_name": "better_name",
      "source_code": "def better_name(...): ...",
      "description": "one sentence"
    }
  ],
  "refactors": [
    {
      "replace_name": "too_specific_name",
      "new_name": "general_name",
      "source_code": "def general_name(...): ...",
      "description": "one sentence"
    }
  ]
}

If there is nothing to merge or refactor, return: {"merges": [], "refactors": []}
"""
```

- [ ] **Step 2: Create `agent/librarian.py`**

```python
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic
import numpy as np

from agent.prompts import LIBRARIAN_SYSTEM_PROMPT
from agent.retriever import embed
from agent.sandbox import run_in_sandbox
from config import Config
from library.tool_library import ToolLibrary

log = logging.getLogger(__name__)

CLUSTER_THRESHOLD = 0.85  # Tools more similar than this get sent to librarian


@dataclass
class LibrarianReport:
    tools_merged: int = 0
    tools_refactored: int = 0
    library_size_before: int = 0
    library_size_after: int = 0
    details: list[str] = field(default_factory=list)


class Librarian:
    def __init__(self, library: ToolLibrary, client: anthropic.Anthropic) -> None:
        self.library = library
        self.client = client

    def run_pass(self) -> LibrarianReport:
        """One librarian pass: cluster → propose merges → validate → replace."""
        report = LibrarianReport()
        all_tools = self.library.get_all()
        report.library_size_before = len(all_tools)

        if len(all_tools) < 2:
            report.library_size_after = len(all_tools)
            return report

        clusters = self._cluster(all_tools)
        log.info("Librarian: found %d clusters with >1 tool", len(clusters))

        for cluster in clusters:
            if Config.ABLATION_NO_LIBRARIAN:
                break
            self._process_cluster(cluster, report)

        report.library_size_after = len(self.library.get_all())
        return report

    def _cluster(self, tools: list[dict]) -> list[list[dict]]:
        """Group tools whose embeddings are more similar than CLUSTER_THRESHOLD."""
        if not tools:
            return []

        # Build embedding matrix
        embeddings = []
        for t in tools:
            emb = np.frombuffer(t["embedding"], dtype=np.float32)
            embeddings.append(emb)
        E = np.array(embeddings)

        # Greedy clustering
        assigned = [False] * len(tools)
        clusters = []
        for i in range(len(tools)):
            if assigned[i]:
                continue
            cluster = [tools[i]]
            assigned[i] = True
            for j in range(i + 1, len(tools)):
                if assigned[j]:
                    continue
                sim = float(np.dot(E[i], E[j]))
                if sim >= CLUSTER_THRESHOLD:
                    cluster.append(tools[j])
                    assigned[j] = True
            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _process_cluster(self, cluster: list[dict], report: LibrarianReport) -> None:
        """Send a cluster to the LLM, validate the result, replace in library."""
        tools_summary = json.dumps(
            [{"name": t["name"], "description": t["description"],
              "source_code": t["source_code"]} for t in cluster],
            indent=2,
        )

        try:
            response = self.client.messages.create(
                model=Config.MODEL,
                max_tokens=2000,
                system=LIBRARIAN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": tools_summary}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            proposal = json.loads(text)
        except Exception as e:
            log.warning("Librarian LLM call failed for cluster: %s", e)
            return

        for merge in proposal.get("merges", []):
            self._apply_merge(merge, report)

        for refactor in proposal.get("refactors", []):
            self._apply_refactor(refactor, report)

    def _apply_merge(self, merge: dict, report: LibrarianReport) -> None:
        source = merge.get("source_code", "")
        new_name = merge.get("new_name", "")
        replace_names = merge.get("replace_names", [])

        # Validate via sandbox with a trivial call
        result = run_in_sandbox(
            source_code=source,
            function_name=new_name,
            args={},
            timeout=Config.VALIDATION_TIMEOUT,
        )
        # We just need it to not crash on import; empty args may fail — that's OK
        # A successful sandbox run OR an arg-error (not a syntax error) is acceptable
        if result.error and "SyntaxError" in (result.error or ""):
            log.warning("Librarian: merged tool %s has syntax error, skipping", new_name)
            return

        # Delete old tools and add merged one
        for name in replace_names:
            self.library.delete_tool(name)

        new_emb = embed(merge.get("description", new_name))
        self.library.add_tool(
            {
                "name": new_name,
                "description": merge.get("description", ""),
                "source_code": source,
                "args": {},
                "returns": "str",
                "tags": [],
            },
            embedding=new_emb,
            task_context="librarian_merge",
        )
        report.tools_merged += len(replace_names)
        report.details.append(f"Merged {replace_names} → {new_name}")
        log.info("Librarian: merged %s → %s", replace_names, new_name)

    def _apply_refactor(self, refactor: dict, report: LibrarianReport) -> None:
        source = refactor.get("source_code", "")
        new_name = refactor.get("new_name", "")
        old_name = refactor.get("replace_name", "")

        result = run_in_sandbox(
            source_code=source,
            function_name=new_name,
            args={},
            timeout=Config.VALIDATION_TIMEOUT,
        )
        if result.error and "SyntaxError" in (result.error or ""):
            log.warning("Librarian: refactored tool %s has syntax error, skipping", new_name)
            return

        new_emb = embed(refactor.get("description", new_name))
        self.library.delete_tool(old_name)
        self.library.add_tool(
            {
                "name": new_name,
                "description": refactor.get("description", ""),
                "source_code": source,
                "args": {},
                "returns": "str",
                "tags": [],
            },
            embedding=new_emb,
            task_context="librarian_refactor",
        )
        report.tools_refactored += 1
        report.details.append(f"Refactored {old_name} → {new_name}")
        log.info("Librarian: refactored %s → %s", old_name, new_name)
```

- [ ] **Step 3: Add `/api/librarian/run` to `web/app.py`**

Add this route to `web/app.py`:

```python
from agent.librarian import Librarian
import anthropic

@app.route("/api/librarian/run", methods=["POST"])
def run_librarian():
    if tool_library is None:
        return jsonify({"error": "Library not initialized"}), 500
    client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    librarian = Librarian(library=tool_library, client=client)
    report = librarian.run_pass()
    return jsonify({
        "tools_merged": report.tools_merged,
        "tools_refactored": report.tools_refactored,
        "library_size_before": report.library_size_before,
        "library_size_after": report.library_size_after,
        "details": report.details,
    })
```

Also add `from config import Config` to the imports in `web/app.py` if not already there.

- [ ] **Step 4: Manual test — run librarian on a small library**

First add a few redundant-ish tools via the agent, then:

```bash
curl -X POST http://localhost:5000/api/librarian/run
```

Expected: JSON with `tools_merged`, `tools_refactored`, `library_size_before/after`.

- [ ] **Step 5: Commit**

```bash
git add agent/librarian.py agent/prompts.py web/app.py
git commit -m "Add Librarian agent with clustering, LLM-propose-merge, and sandbox validation"
```

---

## Task 6: Build Evaluation Task Batches

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/tasks/math_batch1.jsonl`
- Create: `eval/tasks/text_batch2.jsonl`
- Create: `eval/tasks/mixed_batch3.jsonl`

- [ ] **Step 1: Create `eval/__init__.py`**

Empty file.

- [ ] **Step 2: Create `eval/tasks/math_batch1.jsonl`**

50 math tasks. Each line is a JSON object `{"id": "m1_001", "task": "...", "expected": "...", "domain": "math"}`.

Create the file with these 50 tasks (representative sample shown — repeat the pattern for all 50):

```jsonl
{"id": "m1_001", "task": "What is 17 multiplied by 23?", "expected": "391", "domain": "math"}
{"id": "m1_002", "task": "What is the square root of 144?", "expected": "12", "domain": "math"}
{"id": "m1_003", "task": "What is 15% of 240?", "expected": "36", "domain": "math"}
{"id": "m1_004", "task": "What is the sum of the first 10 natural numbers?", "expected": "55", "domain": "math"}
{"id": "m1_005", "task": "Is 97 a prime number?", "expected": "true", "domain": "math"}
{"id": "m1_006", "task": "What is 2 raised to the power of 10?", "expected": "1024", "domain": "math"}
{"id": "m1_007", "task": "What is the GCD of 48 and 18?", "expected": "6", "domain": "math"}
{"id": "m1_008", "task": "What is the LCM of 4 and 6?", "expected": "12", "domain": "math"}
{"id": "m1_009", "task": "What is the factorial of 7?", "expected": "5040", "domain": "math"}
{"id": "m1_010", "task": "What is 100 divided by 8?", "expected": "12.5", "domain": "math"}
{"id": "m1_011", "task": "What is the median of [3, 1, 4, 1, 5, 9, 2, 6]?", "expected": "3.5", "domain": "math"}
{"id": "m1_012", "task": "What is the mean of [10, 20, 30, 40, 50]?", "expected": "30.0", "domain": "math"}
{"id": "m1_013", "task": "What is 7 modulo 3?", "expected": "1", "domain": "math"}
{"id": "m1_014", "task": "What is the 10th Fibonacci number?", "expected": "55", "domain": "math"}
{"id": "m1_015", "task": "Find all prime factors of 84.", "expected": "2, 3, 7", "domain": "math"}
{"id": "m1_016", "task": "What is 3/4 + 1/6 as a fraction?", "expected": "11/12", "domain": "math"}
{"id": "m1_017", "task": "What is the area of a circle with radius 5? (use pi=3.14159)", "expected": "78.53975", "domain": "math"}
{"id": "m1_018", "task": "How many combinations are there in C(10, 3)?", "expected": "120", "domain": "math"}
{"id": "m1_019", "task": "What is the standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]?", "expected": "2.0", "domain": "math"}
{"id": "m1_020", "task": "Convert 255 from decimal to hexadecimal.", "expected": "ff", "domain": "math"}
{"id": "m1_021", "task": "What is log base 2 of 64?", "expected": "6.0", "domain": "math"}
{"id": "m1_022", "task": "What is the hypotenuse of a right triangle with legs 3 and 4?", "expected": "5.0", "domain": "math"}
{"id": "m1_023", "task": "How many integers from 1 to 100 are divisible by both 3 and 5?", "expected": "6", "domain": "math"}
{"id": "m1_024", "task": "What is the perimeter of a rectangle with width 7 and height 4?", "expected": "22", "domain": "math"}
{"id": "m1_025", "task": "What is 1000 in binary?", "expected": "1111101000", "domain": "math"}
{"id": "m1_026", "task": "Sort [5, 3, 8, 1, 9, 2, 7, 4, 6] in ascending order.", "expected": "[1, 2, 3, 4, 5, 6, 7, 8, 9]", "domain": "math"}
{"id": "m1_027", "task": "What is the absolute difference between -17 and 42?", "expected": "59", "domain": "math"}
{"id": "m1_028", "task": "What is 0.1 + 0.2 rounded to 1 decimal place?", "expected": "0.3", "domain": "math"}
{"id": "m1_029", "task": "How many primes are there less than 50?", "expected": "15", "domain": "math"}
{"id": "m1_030", "task": "What is the sum of digits of 9876?", "expected": "30", "domain": "math"}
{"id": "m1_031", "task": "What is 12 choose 5?", "expected": "792", "domain": "math"}
{"id": "m1_032", "task": "What is the ceiling of 4.3?", "expected": "5", "domain": "math"}
{"id": "m1_033", "task": "What is the floor of 7.9?", "expected": "7", "domain": "math"}
{"id": "m1_034", "task": "What is 2^16?", "expected": "65536", "domain": "math"}
{"id": "m1_035", "task": "What is the smallest 4-digit palindrome?", "expected": "1001", "domain": "math"}
{"id": "m1_036", "task": "What is 1/3 as a decimal rounded to 4 places?", "expected": "0.3333", "domain": "math"}
{"id": "m1_037", "task": "What is the number of seconds in a week?", "expected": "604800", "domain": "math"}
{"id": "m1_038", "task": "What is the geometric mean of 4 and 9?", "expected": "6.0", "domain": "math"}
{"id": "m1_039", "task": "What is 36 in octal?", "expected": "44", "domain": "math"}
{"id": "m1_040", "task": "What is the sum of all even numbers from 1 to 100?", "expected": "2550", "domain": "math"}
{"id": "m1_041", "task": "What is the remainder when 157 is divided by 13?", "expected": "1", "domain": "math"}
{"id": "m1_042", "task": "What is the volume of a cube with side 4?", "expected": "64", "domain": "math"}
{"id": "m1_043", "task": "What is the 20th triangular number?", "expected": "210", "domain": "math"}
{"id": "m1_044", "task": "What is the harmonic mean of 1, 2, and 4?", "expected": "1.7143", "domain": "math"}
{"id": "m1_045", "task": "How many perfect squares are between 1 and 200 inclusive?", "expected": "14", "domain": "math"}
{"id": "m1_046", "task": "What is 5! + 4! + 3!?", "expected": "150", "domain": "math"}
{"id": "m1_047", "task": "What is the largest prime below 100?", "expected": "97", "domain": "math"}
{"id": "m1_048", "task": "What is 1000000 in scientific notation?", "expected": "1.0e+06", "domain": "math"}
{"id": "m1_049", "task": "What is the dot product of [1, 2, 3] and [4, 5, 6]?", "expected": "32", "domain": "math"}
{"id": "m1_050", "task": "What is the number of ways to arrange the letters in 'MATH'?", "expected": "24", "domain": "math"}
```

- [ ] **Step 3: Create `eval/tasks/text_batch2.jsonl`**

50 text-processing tasks:

```jsonl
{"id": "t2_001", "task": "Count the number of vowels in 'extraordinary'.", "expected": "6", "domain": "text"}
{"id": "t2_002", "task": "Reverse the string 'hello world'.", "expected": "dlrow olleh", "domain": "text"}
{"id": "t2_003", "task": "How many words are in the sentence 'The quick brown fox jumps over the lazy dog'?", "expected": "9", "domain": "text"}
{"id": "t2_004", "task": "Convert 'hello world' to title case.", "expected": "Hello World", "domain": "text"}
{"id": "t2_005", "task": "Is 'racecar' a palindrome?", "expected": "true", "domain": "text"}
{"id": "t2_006", "task": "Remove all spaces from 'h e l l o  w o r l d'.", "expected": "helloworld", "domain": "text"}
{"id": "t2_007", "task": "Count the occurrences of the letter 'l' in 'hello world'.", "expected": "3", "domain": "text"}
{"id": "t2_008", "task": "What is the longest word in 'The quick brown fox jumps'?", "expected": "quick", "domain": "text"}
{"id": "t2_009", "task": "Extract all email addresses from: 'Contact alice@example.com or bob@test.org for help.'", "expected": "[\"alice@example.com\", \"bob@test.org\"]", "domain": "text"}
{"id": "t2_010", "task": "How many unique characters are in 'mississippi'?", "expected": "4", "domain": "text"}
{"id": "t2_011", "task": "Capitalize the first letter of each sentence in: 'hello world. how are you. fine thanks.'", "expected": "Hello world. How are you. Fine thanks.", "domain": "text"}
{"id": "t2_012", "task": "Replace all spaces with underscores in 'hello world foo bar'.", "expected": "hello_world_foo_bar", "domain": "text"}
{"id": "t2_013", "task": "What is the most frequent word in 'to be or not to be that is the question to be'?", "expected": "to", "domain": "text"}
{"id": "t2_014", "task": "How many sentences are in: 'Hello world. How are you? Fine, thanks! Great.'", "expected": "4", "domain": "text"}
{"id": "t2_015", "task": "Convert 'Hello World' to snake_case.", "expected": "hello_world", "domain": "text"}
{"id": "t2_016", "task": "Does the string 'Python3.11' contain a digit?", "expected": "true", "domain": "text"}
{"id": "t2_017", "task": "What is the 3rd word in 'the quick brown fox'?", "expected": "brown", "domain": "text"}
{"id": "t2_018", "task": "Truncate 'Hello, World!' to 5 characters.", "expected": "Hello", "domain": "text"}
{"id": "t2_019", "task": "How many lines are in a string with 3 newlines?", "expected": "4", "domain": "text"}
{"id": "t2_020", "task": "Extract all numbers from the string 'There are 3 cats and 12 dogs'.", "expected": "[3, 12]", "domain": "text"}
{"id": "t2_021", "task": "Is 'abcde' in alphabetical order?", "expected": "true", "domain": "text"}
{"id": "t2_022", "task": "Count consonants in 'programming'.", "expected": "8", "domain": "text"}
{"id": "t2_023", "task": "What is the character at index 4 in 'python'?", "expected": "o", "domain": "text"}
{"id": "t2_024", "task": "Remove duplicate words from 'the cat sat on the mat the cat'.", "expected": "the cat sat on mat", "domain": "text"}
{"id": "t2_025", "task": "Does 'foobar' start with 'foo'?", "expected": "true", "domain": "text"}
{"id": "t2_026", "task": "Encode 'Hello' in base64.", "expected": "SGVsbG8=", "domain": "text"}
{"id": "t2_027", "task": "What is the MD5 hash of 'hello'?", "expected": "5d41402abc4b2a76b9719d911017c592", "domain": "text"}
{"id": "t2_028", "task": "How many characters are in 'supercalifragilistic'?", "expected": "20", "domain": "text"}
{"id": "t2_029", "task": "Split 'a,b,c,d,e' by comma and return as a list.", "expected": "[\"a\", \"b\", \"c\", \"d\", \"e\"]", "domain": "text"}
{"id": "t2_030", "task": "What index does 'world' start at in 'hello world'?", "expected": "6", "domain": "text"}
{"id": "t2_031", "task": "Is 'level' a palindrome?", "expected": "true", "domain": "text"}
{"id": "t2_032", "task": "Convert 'CamelCaseText' to 'Camel Case Text'.", "expected": "Camel Case Text", "domain": "text"}
{"id": "t2_033", "task": "How many times does 'ab' appear in 'ababab'?", "expected": "3", "domain": "text"}
{"id": "t2_034", "task": "What is the second-to-last character in 'python'?", "expected": "o", "domain": "text"}
{"id": "t2_035", "task": "Strip leading and trailing spaces from '   hello   '.", "expected": "hello", "domain": "text"}
{"id": "t2_036", "task": "Are 'listen' and 'silent' anagrams?", "expected": "true", "domain": "text"}
{"id": "t2_037", "task": "Count words that start with 'p' in 'python is pretty powerful and popular'.", "expected": "4", "domain": "text"}
{"id": "t2_038", "task": "Convert the string '42' to an integer and add 8.", "expected": "50", "domain": "text"}
{"id": "t2_039", "task": "What is the longest common prefix of 'flower', 'flow', 'flight'?", "expected": "fl", "domain": "text"}
{"id": "t2_040", "task": "How many capital letters are in 'Hello World Python'?", "expected": "3", "domain": "text"}
{"id": "t2_041", "task": "Repeat the string 'ab' 4 times.", "expected": "abababab", "domain": "text"}
{"id": "t2_042", "task": "Extract the domain from the URL 'https://www.example.com/path'.", "expected": "www.example.com", "domain": "text"}
{"id": "t2_043", "task": "How many words longer than 4 characters are in 'The quick brown fox jumps over the lazy dog'?", "expected": "4", "domain": "text"}
{"id": "t2_044", "task": "Join ['apple', 'banana', 'cherry'] with ' | '.", "expected": "apple | banana | cherry", "domain": "text"}
{"id": "t2_045", "task": "Is 'Python' a substring of 'I love Python programming'?", "expected": "true", "domain": "text"}
{"id": "t2_046", "task": "Remove all punctuation from 'Hello, World! How are you?'", "expected": "Hello World How are you", "domain": "text"}
{"id": "t2_047", "task": "What is the frequency of each character in 'aab'?", "expected": "{\"a\": 2, \"b\": 1}", "domain": "text"}
{"id": "t2_048", "task": "Pad the string 'hi' to length 10 with dashes on the right.", "expected": "hi--------", "domain": "text"}
{"id": "t2_049", "task": "What is the ROT13 encoding of 'Hello'?", "expected": "Uryyb", "domain": "text"}
{"id": "t2_050", "task": "Find all words with exactly 5 letters in 'The quick brown fox jumps over lazy dogs'.", "expected": "[\"quick\", \"brown\", \"jumps\"]", "domain": "text"}
```

- [ ] **Step 4: Create `eval/tasks/mixed_batch3.jsonl`**

50 cross-domain tasks that require applying math/text/logic tools in combination or slightly different contexts:

```jsonl
{"id": "x3_001", "task": "Count the number of prime digits in 12345.", "expected": "3", "domain": "mixed"}
{"id": "x3_002", "task": "What is the sum of ASCII values of 'Hello'?", "expected": "500", "domain": "mixed"}
{"id": "x3_003", "task": "How many words in 'hello world foo bar' have an even number of characters?", "expected": "2", "domain": "mixed"}
{"id": "x3_004", "task": "What is the product of the digits of 2357?", "expected": "210", "domain": "mixed"}
{"id": "x3_005", "task": "Convert the number 26 to its corresponding uppercase letter (A=1).", "expected": "Z", "domain": "mixed"}
{"id": "x3_006", "task": "How many vowels are in the first 5 Fibonacci numbers written as words (one two three five eight)?", "expected": "9", "domain": "mixed"}
{"id": "x3_007", "task": "What is the length of the longest word that is also a palindrome in 'level noon racecar step'?", "expected": "7", "domain": "mixed"}
{"id": "x3_008", "task": "What is the sum of all digit characters in 'abc123def456'?", "expected": "21", "domain": "mixed"}
{"id": "x3_009", "task": "How many 3-letter words are in 'the cat sat on a mat by the big red bus'?", "expected": "7", "domain": "mixed"}
{"id": "x3_010", "task": "What is the average word length in 'the quick brown fox'?", "expected": "3.75", "domain": "mixed"}
{"id": "x3_011", "task": "Generate the first 8 numbers of the Fibonacci sequence as a comma-separated string.", "expected": "1, 1, 2, 3, 5, 8, 13, 21", "domain": "mixed"}
{"id": "x3_012", "task": "How many letters in the English alphabet are also valid Roman numeral characters (I, V, X, L, C, D, M)?", "expected": "7", "domain": "mixed"}
{"id": "x3_013", "task": "What is the numeric value of the string '42abc' (extract and sum all numbers)?", "expected": "42", "domain": "mixed"}
{"id": "x3_014", "task": "Is the number of characters in 'hello' a prime?", "expected": "true", "domain": "mixed"}
{"id": "x3_015", "task": "What is the most common digit in the first 20 Fibonacci numbers?", "expected": "1", "domain": "mixed"}
{"id": "x3_016", "task": "Encode the number 255 as a zero-padded 8-character binary string.", "expected": "11111111", "domain": "mixed"}
{"id": "x3_017", "task": "How many characters does it take to spell out the number 17 ('seventeen')?", "expected": "9", "domain": "mixed"}
{"id": "x3_018", "task": "What is the checksum (sum of digits) of the year 2025?", "expected": "9", "domain": "mixed"}
{"id": "x3_019", "task": "Sort the words in 'banana apple cherry date' alphabetically and return them joined by space.", "expected": "apple banana cherry date", "domain": "mixed"}
{"id": "x3_020", "task": "What is the lexicographic rank of 'b' among the unique characters in 'abcde'?", "expected": "2", "domain": "mixed"}
{"id": "x3_021", "task": "How many characters are in the string representation of 2^32?", "expected": "10", "domain": "mixed"}
{"id": "x3_022", "task": "What is the sum of the positions (1-indexed) of all vowels in 'hello'?", "expected": "10", "domain": "mixed"}
{"id": "x3_023", "task": "How many prime numbers have exactly 2 digits?", "expected": "21", "domain": "mixed"}
{"id": "x3_024", "task": "Convert 'abc' to its numeric representation where a=1, b=2, c=3, etc. and return the sum.", "expected": "6", "domain": "mixed"}
{"id": "x3_025", "task": "What is the product of the lengths of the words in 'cat is fun'?", "expected": "27", "domain": "mixed"}
{"id": "x3_026", "task": "How many even numbers are in the first 15 Fibonacci numbers?", "expected": "5", "domain": "mixed"}
{"id": "x3_027", "task": "What is the longest word in a sentence with 5 words where each word length equals its position?", "expected": "5", "domain": "mixed"}
{"id": "x3_028", "task": "What is the square root of the number of letters in 'uncopyrightable' (15 letters)?", "expected": "3.872983", "domain": "mixed"}
{"id": "x3_029", "task": "How many numbers between 1 and 100 are both perfect squares and have a digit sum divisible by 3?", "expected": "3", "domain": "mixed"}
{"id": "x3_030", "task": "Interleave the strings 'abc' and '123' character by character.", "expected": "a1b2c3", "domain": "mixed"}
{"id": "x3_031", "task": "What is the number of distinct letters in 'mississippi'?", "expected": "4", "domain": "mixed"}
{"id": "x3_032", "task": "What Roman numeral represents 42?", "expected": "XLII", "domain": "mixed"}
{"id": "x3_033", "task": "How many numbers from 1 to 1000 contain the digit 7?", "expected": "271", "domain": "mixed"}
{"id": "x3_034", "task": "What is the number of words in the sentence formed by the first letter of each Fibonacci number up to 21?", "expected": "1", "domain": "mixed"}
{"id": "x3_035", "task": "Compute the sum of ASCII values of all uppercase letters in 'Hello World'.", "expected": "159", "domain": "mixed"}
{"id": "x3_036", "task": "What is the largest palindrome number less than 1000?", "expected": "999", "domain": "mixed"}
{"id": "x3_037", "task": "How many words in 'one two three four five six seven eight nine ten' have a prime number of letters?", "expected": "5", "domain": "mixed"}
{"id": "x3_038", "task": "Generate a comma-separated list of the first 5 perfect squares.", "expected": "1, 4, 9, 16, 25", "domain": "mixed"}
{"id": "x3_039", "task": "What is the number of characters in the hex representation of 255 (without '0x' prefix)?", "expected": "2", "domain": "mixed"}
{"id": "x3_040", "task": "Is the string 'A man a plan a canal Panama' a palindrome when ignoring case and spaces?", "expected": "true", "domain": "mixed"}
{"id": "x3_041", "task": "How many numbers from 1 to 50 are divisible by 3 or 5 but not both?", "expected": "23", "domain": "mixed"}
{"id": "x3_042", "task": "What is the Caesar cipher (shift=3) of 'ABC'?", "expected": "DEF", "domain": "mixed"}
{"id": "x3_043", "task": "How many times does the digit 1 appear in numbers from 1 to 20?", "expected": "12", "domain": "mixed"}
{"id": "x3_044", "task": "What is the sum of all 2-digit palindromes?", "expected": "495", "domain": "mixed"}
{"id": "x3_045", "task": "Find the first repeated character in 'abcdebc'.", "expected": "b", "domain": "mixed"}
{"id": "x3_046", "task": "How many 3-digit numbers are perfect cubes?", "expected": "4", "domain": "mixed"}
{"id": "x3_047", "task": "What is the Collatz sequence length starting from 27?", "expected": "112", "domain": "mixed"}
{"id": "x3_048", "task": "How many letters in 'Hello World' are in the second half of the alphabet (N-Z)?", "expected": "3", "domain": "mixed"}
{"id": "x3_049", "task": "What is the number of syllables in 'extraordinary' (count vowel groups)?", "expected": "6", "domain": "mixed"}
{"id": "x3_050", "task": "What is the sum of the first 10 square numbers?", "expected": "385", "domain": "mixed"}
```

- [ ] **Step 5: Commit**

```bash
git add eval/__init__.py eval/tasks/
git commit -m "Add evaluation task batches: 50 math, 50 text, 50 mixed cross-domain tasks"
```

---

## Task 7: Build Evaluation Harness (`run_eval.py`)

**Files:**
- Create: `eval/run_eval.py`

- [ ] **Step 1: Create `eval/run_eval.py`**

```python
"""
Evaluation harness for ToolForge.

Usage:
    python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition full
    python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition no_library
    python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition no_abstraction
    python -m eval.run_eval --batch eval/tasks/text_batch2.jsonl --condition full
    python -m eval.run_eval --batch eval/tasks/mixed_batch3.jsonl --condition full

Conditions:
    full            - Full system (library + abstraction prompt + librarian)
    no_library      - No tool library (fresh LLM call every task)
    no_abstraction  - Library but no abstraction prompt in tool creator
    baseline        - Alias for no_library

Results are written to eval/results/<batch>_<condition>_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from library.tool_library import ToolLibrary
from agent.loop import AgentLoop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_tasks(path: str) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def answers_match(actual: str, expected: str) -> bool:
    """Loose comparison: strip, lowercase, try numeric equality."""
    a = actual.strip().lower()
    e = expected.strip().lower()
    if a == e:
        return True
    # Try numeric comparison
    try:
        return abs(float(a) - float(e)) < 1e-3
    except ValueError:
        pass
    # Try extracting first number from actual
    import re
    nums_a = re.findall(r"-?\d+\.?\d*", a)
    nums_e = re.findall(r"-?\d+\.?\d*", e)
    if nums_a and nums_e:
        try:
            return abs(float(nums_a[0]) - float(nums_e[0])) < 1e-3
        except ValueError:
            pass
    return False


def run_eval(batch_path: str, condition: str) -> str:
    """Run evaluation and return path to CSV output file."""
    # Apply ablation flags via environment
    if condition in ("no_library", "baseline"):
        os.environ["ABLATION_NO_LIBRARY"] = "true"
    elif condition == "no_abstraction":
        os.environ["ABLATION_NO_ABSTRACTION"] = "true"
    else:  # full
        os.environ.pop("ABLATION_NO_LIBRARY", None)
        os.environ.pop("ABLATION_NO_ABSTRACTION", None)
        os.environ.pop("ABLATION_NO_LIBRARIAN", None)

    tasks = load_tasks(batch_path)
    batch_name = os.path.splitext(os.path.basename(batch_path))[0]

    os.makedirs("eval/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"eval/results/{batch_name}_{condition}_{timestamp}.csv"

    # Use a fresh library per run for clean ablation
    db_path = f"eval/results/tools_{batch_name}_{condition}_{timestamp}.db"
    library = ToolLibrary(db_path=db_path, seed=True)

    fieldnames = [
        "task_id", "domain", "task", "expected", "actual",
        "success", "tool_created", "tool_reused", "tool_name",
        "attempts", "time_seconds", "condition",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, t in enumerate(tasks):
            log.info("[%d/%d] %s: %s", i + 1, len(tasks), t["id"], t["task"][:60])
            start = time.time()

            try:
                loop = AgentLoop(tool_library=library)
                result = loop.run(t["task"])
                elapsed = time.time() - start

                actual = result.get("answer", "") or ""
                success = result.get("success", False) and answers_match(actual, t["expected"])

                writer.writerow({
                    "task_id": t["id"],
                    "domain": t.get("domain", ""),
                    "task": t["task"],
                    "expected": t["expected"],
                    "actual": actual,
                    "success": success,
                    "tool_created": result.get("tool_created", False),
                    "tool_reused": result.get("tool_reused", False),
                    "tool_name": result.get("tool_used", ""),
                    "attempts": result.get("attempts", 1),
                    "time_seconds": round(elapsed, 2),
                    "condition": condition,
                })
                csvfile.flush()

            except Exception as e:
                log.error("Task %s failed with exception: %s", t["id"], e)
                writer.writerow({
                    "task_id": t["id"], "domain": t.get("domain", ""), "task": t["task"],
                    "expected": t["expected"], "actual": "", "success": False,
                    "tool_created": False, "tool_reused": False, "tool_name": "",
                    "attempts": 0, "time_seconds": 0, "condition": condition,
                })
                csvfile.flush()

    log.info("Results written to %s", out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="Path to .jsonl task file")
    parser.add_argument(
        "--condition",
        choices=["full", "no_library", "no_abstraction", "baseline"],
        default="full",
    )
    args = parser.parse_args()
    run_eval(args.batch, args.condition)
```

Also update `agent/loop.py` to return `tool_created` and `tool_reused` in the result dict:

In `AgentLoop.run()`, track these in instance variables and include in the return dict:
```python
self._tool_created = False
self._tool_reused = False
self._tool_name_used = ""
# ... in _handle_call_tool: self._tool_reused = True; self._tool_name_used = tool_name
# ... in _handle_acquire_tool after success: self._tool_created = True; self._tool_name_used = spec["name"]
# ... in final_answer return:
return {
    "success": True, "answer": answer,
    "tool_created": self._tool_created,
    "tool_reused": self._tool_reused,
    "tool_used": self._tool_name_used,
    "attempts": self.iteration,
}
```

- [ ] **Step 2: Run Batch 1 (math) under full condition as a smoke test with 5 tasks**

Edit `math_batch1.jsonl` to temporarily only have 5 lines, run:

```bash
python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition full
```

Expected: CSV created in `eval/results/`. Open it and verify `success` column has some `True` rows.

- [ ] **Step 3: Restore full task batches and commit**

```bash
git add eval/run_eval.py agent/loop.py
git commit -m "Add evaluation harness with 4-condition ablation support"
```

---

## Task 8: Build Analysis & Plotting (`analyze.py`)

**Files:**
- Create: `eval/analyze.py`

- [ ] **Step 1: Create `eval/analyze.py`**

```python
"""
Analyze evaluation results and produce plots.

Usage:
    python -m eval.analyze --results eval/results/
    python -m eval.analyze --results eval/results/ --out eval/plots/
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_all_results(results_dir: str) -> pd.DataFrame:
    dfs = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(results_dir, fname))
            dfs.append(df)
    if not dfs:
        raise ValueError(f"No CSV files found in {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 6 key metrics per (condition, domain, batch) group."""
    results = []
    for (condition, domain), g in df.groupby(["condition", "domain"]):
        n = len(g)
        n_success = g["success"].sum()
        n_created = g["tool_created"].sum()
        n_reused = g["tool_reused"].sum()

        results.append({
            "condition": condition,
            "domain": domain,
            "n_tasks": n,
            "task_success_rate": n_success / n if n else 0,
            "tool_creation_rate": n_created / n if n else 0,
            "tool_reuse_rate": n_reused / max(n_reused + n_created, 1),
            "avg_time": g["time_seconds"].mean(),
            "avg_attempts": g["attempts"].mean(),
        })
    return pd.DataFrame(results)


def plot_success_rate(metrics: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = metrics.pivot_table(index="domain", columns="condition", values="task_success_rate")
    data.plot(kind="bar", ax=ax)
    ax.set_title("Task Success Rate by Domain and Condition")
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Domain")
    ax.set_ylim(0, 1)
    ax.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "success_rate.png"), dpi=150)
    plt.close()


def plot_reuse_rate_over_time(df: pd.DataFrame, out_dir: str) -> None:
    """Reuse rate as tasks accumulate — does it increase as library matures?"""
    full = df[df["condition"] == "full"].copy()
    full = full.sort_values("task_id")
    full["cumulative_reuse"] = (full["tool_reused"].cumsum() /
                                 (full["tool_reused"] + full["tool_created"]).cumsum().clip(lower=1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(full)), full["cumulative_reuse"])
    ax.set_title("Cumulative Tool Reuse Rate Over Time (Full Condition)")
    ax.set_xlabel("Task Number")
    ax.set_ylabel("Cumulative Reuse Rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reuse_over_time.png"), dpi=150)
    plt.close()


def plot_cross_domain_transfer(metrics: pd.DataFrame, out_dir: str) -> None:
    """Compare math vs text vs mixed success rate (H1: cross-domain generalization)."""
    full_only = metrics[metrics["condition"] == "full"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=full_only, x="domain", y="task_success_rate", ax=ax)
    ax.set_title("Cross-Domain Transfer: Success Rate by Domain (Full Condition)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cross_domain.png"), dpi=150)
    plt.close()


def plot_ablation_comparison(metrics: pd.DataFrame, out_dir: str) -> None:
    """H2 + H3: Does abstraction prompt / librarian matter?"""
    fig, ax = plt.subplots(figsize=(10, 6))
    agg = metrics.groupby("condition")["task_success_rate"].mean().reset_index()
    sns.barplot(data=agg, x="condition", y="task_success_rate", ax=ax)
    ax.set_title("Ablation Study: Overall Success Rate by Condition")
    ax.set_ylabel("Mean Task Success Rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation.png"), dpi=150)
    plt.close()


def analyze(results_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = load_all_results(results_dir)
    metrics = compute_metrics(df)

    print("\n=== Metrics Summary ===")
    print(metrics.to_string(index=False))

    metrics.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    plot_success_rate(metrics, out_dir)
    plot_reuse_rate_over_time(df, out_dir)
    plot_cross_domain_transfer(metrics, out_dir)
    plot_ablation_comparison(metrics, out_dir)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval/results/", help="Directory with CSV results")
    parser.add_argument("--out", default="eval/plots/", help="Output directory for plots")
    args = parser.parse_args()
    analyze(args.results, args.out)
```

- [ ] **Step 2: Run analyze on whatever results exist from smoke test**

```bash
python -m eval.analyze --results eval/results/ --out eval/plots/
```

Expected: prints metrics table, writes 4 PNG plots to `eval/plots/`.

- [ ] **Step 3: Commit**

```bash
git add eval/analyze.py
git commit -m "Add analysis harness with 4 metric plots for ablation study"
```

---

## Task 9: Write Unit Tests for Remaining Modules

**Files:**
- Create: `tests/test_sandbox.py`
- Create: `tests/test_tool_generator.py` (mocked LLM)

- [ ] **Step 1: Create `tests/test_sandbox.py`**

```python
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


def test_returns_string_for_non_string():
    result = run_in_sandbox(
        source_code="def get_list() -> list:\n    return [1, 2, 3]",
        function_name="get_list",
        args={},
        timeout=5,
    )
    assert result.success
    assert result.result == "[1, 2, 3]"
```

- [ ] **Step 2: Create `tests/test_tool_generator.py`**

```python
import pytest
from unittest.mock import MagicMock, patch
from agent.tool_generator import ToolGenerator


VALID_SPEC = {
    "function_name": "add_numbers",
    "name": "add_numbers",
    "source_code": "def add_numbers(a: int, b: int) -> int:\n    return a + b",
    "description": "Add two integers",
    "args": {"a": "int", "b": "int"},
    "returns": "int",
    "tags": ["math"],
    "test_call": {"a": 1, "b": 2},
}


@pytest.fixture
def mock_generator():
    gen = ToolGenerator.__new__(ToolGenerator)
    gen.client = MagicMock()
    return gen


def test_generate_returns_spec_on_valid_llm_response(mock_generator):
    import json
    mock_generator.client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=json.dumps(VALID_SPEC))]
    )
    result = mock_generator.generate(
        capability_needed="add numbers",
        capability_detail="add two integers a and b",
        task_context="compute 2+3",
    )
    assert result is not None
    assert result["name"] == "add_numbers"


def test_generate_returns_none_on_json_parse_failure(mock_generator):
    mock_generator.client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="this is not json")]
    )
    result = mock_generator.generate(
        capability_needed="add numbers",
        capability_detail="add two integers",
        task_context="compute 2+3",
    )
    assert result is None


def test_generate_includes_failures_in_prompt(mock_generator):
    import json
    mock_generator.client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=json.dumps(VALID_SPEC))]
    )
    failures = [{"attempted_code": "def bad(): pass", "error_msg": "TypeError"}]
    mock_generator.generate(
        capability_needed="add numbers",
        capability_detail="add two integers",
        task_context="compute 2+3",
        failures=failures,
    )
    call_args = mock_generator.client.messages.create.call_args
    user_msg = call_args[1]["messages"][0]["content"]
    assert "TypeError" in user_msg
```

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass. Check coverage:

```bash
pytest tests/ --cov=agent --cov=library --cov-report=term-missing
```

Target: >80% line coverage on `agent/` and `library/`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sandbox.py tests/test_tool_generator.py
git commit -m "Add unit tests for sandbox and tool generator; full test suite passes"
```

---

## Task 10: Run Full Evaluation & Produce Results

**This task runs the actual research experiment. Budget ~$5–10 in API costs.**

- [ ] **Step 1: Run Batch 1 (math) under all 4 conditions**

```bash
# Full system
python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition full

# No library (baseline)
python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition no_library

# No abstraction prompt
python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition no_abstraction
```

Review the `eval/results/` CSVs after Batch 1 math before continuing. Fix any systematic failures (e.g., answer matching too strict).

- [ ] **Step 2: Run Batch 2 (text) under all conditions**

```bash
python -m eval.run_eval --batch eval/tasks/text_batch2.jsonl --condition full
python -m eval.run_eval --batch eval/tasks/text_batch2.jsonl --condition no_library
python -m eval.run_eval --batch eval/tasks/text_batch2.jsonl --condition no_abstraction
```

- [ ] **Step 3: Run Batch 3 (mixed cross-domain) under full condition**

```bash
python -m eval.run_eval --batch eval/tasks/mixed_batch3.jsonl --condition full
```

This is the key batch for H1 (cross-domain generalization).

- [ ] **Step 4: Run the librarian and measure library compression**

After running all batches under `full` condition:

```bash
python -c "
from config import Config
from library.tool_library import ToolLibrary
from agent.librarian import Librarian
import anthropic

lib = ToolLibrary(db_path='eval/results/tools_full_latest.db')
client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
librarian = Librarian(lib, client)
report = librarian.run_pass()
print(report)
"
```

Log the before/after library size.

- [ ] **Step 5: Generate all plots**

```bash
python -m eval.analyze --results eval/results/ --out eval/plots/
```

Review: `success_rate.png`, `reuse_over_time.png`, `cross_domain.png`, `ablation.png`.

- [ ] **Step 6: Final commit**

```bash
git add eval/results/.gitkeep eval/plots/.gitkeep
git commit -m "Evaluation complete: results and plots for all 4 ablation conditions"
```

---

## Self-Review

### Spec Coverage

| Hypothesis | Task(s) that test it |
|-----------|---------------------|
| H1: cross-domain transfer | Task 10 Batch 3 mixed tasks; `cross_domain.png` |
| H2: abstraction prompt matters | Task 10 `no_abstraction` vs `full`; `ablation.png` |
| H3: librarian improves reuse | Task 10 librarian run + library size delta |
| H4: reuse rate increases over time | `reuse_over_time.png` from Task 8 |
| Tool creation loop | Tasks 1–4 (library + retriever + failure store + loop) |
| Sandbox validation | Task 1 (ToolLibrary) + Task 9 (sandbox tests) |
| Semantic retrieval | Task 2 (retriever) |
| Failure injection | Task 3 (failure store) |
| Librarian agent | Task 5 |
| Evaluation harness | Tasks 6, 7, 8, 10 |

All system design components accounted for.

### Placeholder Scan

No TBDs. All code blocks are complete. Seed tools copied verbatim from existing `tool_library.py`. All test cases use real assertions with actual expected values.

### Type Consistency

- `ToolLibrary.add_tool(spec, embedding, task_context)` — used consistently in Tasks 1, 3, 5
- `embed(text) -> np.ndarray` — defined in `agent/retriever.py`, imported in `agent/loop.py`, `agent/librarian.py`, `scripts/reseed_embeddings.py`
- `run_in_sandbox(source_code, function_name, args, timeout) -> SandboxResult` — unchanged from existing code; used in Tasks 4, 5, 9

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-15-toolforge-complete.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
