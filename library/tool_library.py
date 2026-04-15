from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


class ToolLibrary:
    def __init__(self, path: str) -> None:
        self.path = path
        self.tools: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tools = data.get("tools", {})
        else:
            self.tools = {}
            self._seed()
            self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "tools": self.tools}, f, indent=2)

    def _seed(self) -> None:
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
                "tags": ["web", "http", "fetch", "html", "url", "scrape"],
            },
            {
                "name": "run_python_expression",
                "description": "Evaluate a Python expression and return the result (math, logic, string ops)",
                "source_code": (
                    "def run_python_expression(expression: str) -> str:\n"
                    '    """Safely evaluate a Python expression (math, string ops, etc)."""\n'
                    "    import math\n"
                    "    allowed = {'__builtins__': {}, 'math': math, 'abs': abs, 'round': round,\n"
                    "               'min': min, 'max': max, 'sum': sum, 'len': len,\n"
                    "               'int': int, 'float': float, 'str': str, 'list': list,\n"
                    "               'sorted': sorted, 'range': range, 'enumerate': enumerate,\n"
                    "               'zip': zip, 'map': map, 'filter': filter, 'True': True,\n"
                    "               'False': False, 'None': None}\n"
                    "    try:\n"
                    "        result = eval(expression, allowed)\n"
                    "        return str(result)\n"
                    "    except Exception as e:\n"
                    "        return f'Error evaluating expression: {e}'\n"
                ),
                "args": {"expression": "str — Python expression to evaluate"},
                "returns": "str — result of the expression or error message",
                "tags": ["math", "calculate", "eval", "expression", "logic", "compute"],
            },
            {
                "name": "search_text",
                "description": "Search text using a regex pattern and return all matches",
                "source_code": (
                    "def search_text(text: str, pattern: str) -> str:\n"
                    '    """Search text with a regex pattern, return JSON list of matches."""\n'
                    "    import re\n"
                    "    import json\n"
                    "    try:\n"
                    "        matches = re.findall(pattern, text)\n"
                    "        return json.dumps(matches[:100])\n"
                    "    except Exception as e:\n"
                    "        return f'Error in regex search: {e}'\n"
                ),
                "args": {
                    "text": "str — text to search in",
                    "pattern": "str — regex pattern",
                },
                "returns": "str — JSON array of matches (max 100)",
                "tags": ["regex", "search", "text", "pattern", "find", "match"],
            },
        ]

        for tool in seeds:
            tool["created_at"] = datetime.now(timezone.utc).isoformat()
            tool["created_for_task"] = "seed"
            tool["use_count"] = 0
            tool["validated"] = True
            self.tools[tool["name"]] = tool

    def add_tool(self, tool_spec: dict[str, Any], task_context: str = "") -> None:
        tool_spec["created_at"] = datetime.now(timezone.utc).isoformat()
        tool_spec["created_for_task"] = task_context
        tool_spec["use_count"] = 0
        tool_spec["validated"] = True
        self.tools[tool_spec["name"]] = tool_spec
        self._save()

    def increment_use(self, name: str) -> None:
        if name in self.tools:
            self.tools[name]["use_count"] += 1
            self.tools[name]["last_used"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def find_relevant_tools(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())
        scored: list[tuple[float, dict[str, Any]]] = []

        for tool in self.tools.values():
            score = 0.0
            for tag in tool.get("tags", []):
                if tag.lower() in query_words:
                    score += 3
            desc_words = set(tool.get("description", "").lower().split())
            score += len(query_words & desc_words)
            score += min(tool.get("use_count", 0), 5) * 0.5
            if score > 0:
                scored.append((score, tool))

        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:top_k]]

    def get_all_tools_for_prompt(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "args": t["args"],
                "returns": t["returns"],
            }
            for t in self.tools.values()
        ]

    def get_source_code(self, name: str) -> str | None:
        tool = self.tools.get(name)
        return tool["source_code"] if tool else None

    def get_all_tools_public(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "args": t["args"],
                "returns": t["returns"],
                "tags": t.get("tags", []),
                "use_count": t.get("use_count", 0),
                "created_at": t.get("created_at", ""),
                "created_for_task": t.get("created_for_task", ""),
            }
            for t in sorted(
                self.tools.values(), key=lambda x: -x.get("use_count", 0)
            )
        ]
