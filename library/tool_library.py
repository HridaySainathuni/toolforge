from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
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
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
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

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tools (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    name         TEXT NOT NULL UNIQUE,
                    description  TEXT NOT NULL DEFAULT '',
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
        embedding: "np.ndarray",
        task_context: str = "",
    ) -> int:
        """Insert or update a tool, preserving reuse_count/pass_rate on conflict. Returns the row id."""
        vec = self._normalize(embedding.astype(np.float32))
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO tools
                   (name, description, source_code, embedding, args, returns, tags, origin_task)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tool_spec["name"],
                    tool_spec.get("description", ""),
                    tool_spec["source_code"],
                    vec.tobytes(),
                    json.dumps(tool_spec.get("args", {})),
                    tool_spec.get("returns", ""),
                    json.dumps(tool_spec.get("tags", [])),
                    task_context,
                ),
            )
            conn.execute(
                """UPDATE tools SET description=?, source_code=?, embedding=?, args=?, returns=?, tags=?
                   WHERE name=?""",
                (
                    tool_spec.get("description", ""),
                    tool_spec["source_code"],
                    vec.tobytes(),
                    json.dumps(tool_spec.get("args", {})),
                    tool_spec.get("returns", ""),
                    json.dumps(tool_spec.get("tags", [])),
                    tool_spec["name"],
                ),
            )
            row = conn.execute("SELECT id FROM tools WHERE name=?", (tool_spec["name"],)).fetchone()
            return row["id"]

    def replace_tool(self, name: str, new_source: str, new_embedding: "np.ndarray") -> None:
        vec = self._normalize(new_embedding.astype(np.float32))
        with self._conn() as conn:
            conn.execute(
                "UPDATE tools SET source_code=?, embedding=? WHERE name=?",
                (new_source, vec.tobytes(), name),
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
                n = max(row["reuse_count"], 1)
                new_rate = (row["pass_rate"] * n + (1.0 if success else 0.0)) / (n + 1)
                conn.execute(
                    "UPDATE tools SET pass_rate=? WHERE name=?", (new_rate, name)
                )

    def record_outcome(self, name: str, success: bool) -> None:
        """Atomically increment reuse_count and update pass_rate."""
        delta = 1.0 if success else 0.0
        with self._conn() as conn:
            conn.execute(
                """UPDATE tools SET
                   reuse_count = reuse_count + 1,
                   pass_rate = (pass_rate * reuse_count + ?) / (reuse_count + 1)
                   WHERE name=?""",
                (delta, name),
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
        query_embedding: "np.ndarray",
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
        """Return all tools including source code and raw embedding bytes (for librarian)."""
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
        """Seed the library with 5 starter tools using zero embeddings."""
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
        spreadsheet_seeds = [
            {
                "name": "read_spreadsheet",
                "description": "Read an Excel (.xlsx/.xls) or CSV file and return its contents as a markdown table",
                "source_code": (
                    "def read_spreadsheet(path: str, sheet: str = None, max_rows: int = 50) -> str:\n"
                    '    """Read Excel or CSV file, return as markdown table (up to max_rows rows)."""\n'
                    "    import pandas as pd\n"
                    "    try:\n"
                    "        if path.endswith('.csv'):\n"
                    "            df = pd.read_csv(path, nrows=max_rows)\n"
                    "        else:\n"
                    "            df = pd.read_excel(path, sheet_name=sheet or 0, nrows=max_rows, engine='openpyxl')\n"
                    "        return df.to_markdown(index=False)\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {
                    "path": "str — path to .xlsx, .xls, or .csv file",
                    "sheet": "str — sheet name (Excel only, optional)",
                    "max_rows": "int — max rows to return (default 50)",
                },
                "returns": "str — markdown table or error",
                "tags": ["excel", "csv", "spreadsheet", "pandas", "read", "data"],
            },
            {
                "name": "query_spreadsheet",
                "description": "Load a spreadsheet and run a pandas query/expression to filter or aggregate data",
                "source_code": (
                    "def query_spreadsheet(path: str, query: str, sheet: str = None) -> str:\n"
                    '    """Load spreadsheet and evaluate a pandas query string (e.g. \'Age > 30\')."""\n'
                    "    import pandas as pd\n"
                    "    try:\n"
                    "        if path.endswith('.csv'):\n"
                    "            df = pd.read_csv(path)\n"
                    "        else:\n"
                    "            df = pd.read_excel(path, sheet_name=sheet or 0, engine='openpyxl')\n"
                    "        result = df.query(query)\n"
                    "        return result.to_markdown(index=False)\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {
                    "path": "str — path to file",
                    "query": "str — pandas query string e.g. 'column > 100'",
                    "sheet": "str — sheet name (Excel only, optional)",
                },
                "returns": "str — filtered rows as markdown table or error",
                "tags": ["excel", "csv", "spreadsheet", "pandas", "query", "filter"],
            },
            {
                "name": "summarize_spreadsheet",
                "description": "Return shape, column names, dtypes, and summary stats for a spreadsheet",
                "source_code": (
                    "def summarize_spreadsheet(path: str, sheet: str = None) -> str:\n"
                    '    """Return schema and summary stats for an Excel or CSV file."""\n'
                    "    import pandas as pd, json\n"
                    "    try:\n"
                    "        if path.endswith('.csv'):\n"
                    "            df = pd.read_csv(path)\n"
                    "        else:\n"
                    "            df = pd.read_excel(path, sheet_name=sheet or 0, engine='openpyxl')\n"
                    "        info = {\n"
                    "            'rows': len(df), 'columns': len(df.columns),\n"
                    "            'column_names': list(df.columns),\n"
                    "            'dtypes': {c: str(t) for c, t in df.dtypes.items()},\n"
                    "            'null_counts': df.isnull().sum().to_dict(),\n"
                    "            'numeric_summary': json.loads(df.describe().to_json()),\n"
                    "        }\n"
                    "        return json.dumps(info, default=str)\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {
                    "path": "str — path to .xlsx, .xls, or .csv file",
                    "sheet": "str — sheet name (Excel only, optional)",
                },
                "returns": "str — JSON with shape, columns, dtypes, null counts, numeric stats",
                "tags": ["excel", "csv", "spreadsheet", "pandas", "summary", "stats", "schema"],
            },
            {
                "name": "write_spreadsheet",
                "description": "Write a list of dicts or a CSV string to an Excel or CSV file",
                "source_code": (
                    "def write_spreadsheet(path: str, data: list, sheet: str = 'Sheet1') -> str:\n"
                    '    """Write list of dicts to Excel or CSV. data = [{"col": val, ...}, ...]."""\n'
                    "    import pandas as pd\n"
                    "    try:\n"
                    "        df = pd.DataFrame(data)\n"
                    "        if path.endswith('.csv'):\n"
                    "            df.to_csv(path, index=False)\n"
                    "        else:\n"
                    "            df.to_excel(path, sheet_name=sheet, index=False, engine='openpyxl')\n"
                    "        return f'Written {len(df)} rows x {len(df.columns)} cols to {path}'\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {
                    "path": "str — output path (.xlsx or .csv)",
                    "data": "list — list of dicts, each dict is a row",
                    "sheet": "str — sheet name for Excel (default Sheet1)",
                },
                "returns": "str — success message or error",
                "tags": ["excel", "csv", "spreadsheet", "pandas", "write", "export"],
            },
        ]

        file_seeds = [
            {
                "name": "list_directory",
                "description": "List files and directories at a given path",
                "source_code": (
                    "def list_directory(path: str = '.') -> str:\n"
                    '    """List files and subdirectories at path, returns JSON."""\n'
                    "    import os, json\n"
                    "    try:\n"
                    "        entries = []\n"
                    "        for name in sorted(os.listdir(path)):\n"
                    "            full = os.path.join(path, name)\n"
                    "            entries.append({'name': name, 'type': 'dir' if os.path.isdir(full) else 'file',\n"
                    "                            'size': os.path.getsize(full) if os.path.isfile(full) else None})\n"
                    "        return json.dumps(entries)\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {"path": "str — directory path, defaults to '.'"},
                "returns": "str — JSON array of {name, type, size}",
                "tags": ["file", "directory", "list", "ls"],
            },
            {
                "name": "edit_file_replace",
                "description": "Replace all occurrences of a string in a file and save it",
                "source_code": (
                    "def edit_file_replace(path: str, old_text: str, new_text: str) -> str:\n"
                    '    """Replace all occurrences of old_text with new_text in file at path."""\n'
                    "    try:\n"
                    "        with open(path, 'r', encoding='utf-8') as f:\n"
                    "            content = f.read()\n"
                    "        count = content.count(old_text)\n"
                    "        if count == 0:\n"
                    "            return f'No occurrences of the target text found in {path}'\n"
                    "        content = content.replace(old_text, new_text)\n"
                    "        with open(path, 'w', encoding='utf-8') as f:\n"
                    "            f.write(content)\n"
                    "        return f'Replaced {count} occurrence(s) in {path}'\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {
                    "path": "str — file path",
                    "old_text": "str — text to find",
                    "new_text": "str — replacement text",
                },
                "returns": "str — success message with count, or error",
                "tags": ["file", "edit", "replace", "text", "modify"],
            },
            {
                "name": "read_file_lines",
                "description": "Read specific line range from a file",
                "source_code": (
                    "def read_file_lines(path: str, start: int = 1, end: int = 50) -> str:\n"
                    '    """Read lines start..end (1-indexed, inclusive) from file."""\n'
                    "    try:\n"
                    "        with open(path, 'r', encoding='utf-8') as f:\n"
                    "            lines = f.readlines()\n"
                    "        selected = lines[start-1:end]\n"
                    "        return ''.join(f'{start+i}: {l}' for i, l in enumerate(selected))\n"
                    "    except Exception as e:\n"
                    "        return f'Error: {e}'\n"
                ),
                "args": {
                    "path": "str — file path",
                    "start": "int — first line (1-indexed)",
                    "end": "int — last line (inclusive)",
                },
                "returns": "str — numbered lines",
                "tags": ["file", "read", "lines", "range"],
            },
        ]
        for spec in seeds + spreadsheet_seeds + file_seeds:
            self.add_tool(spec, embedding=zero_emb, task_context="seed")
