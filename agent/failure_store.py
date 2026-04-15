from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any


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
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
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
