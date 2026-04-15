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

_CLUSTER_THRESHOLD = 0.85


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
        """One librarian pass: cluster similar tools → propose merges → validate → replace."""
        report = LibrarianReport()

        if Config.ABLATION_NO_LIBRARIAN:
            log.info("Librarian: disabled by ABLATION_NO_LIBRARIAN flag")
            all_tools = self.library.get_all()
            report.library_size_before = len(all_tools)
            report.library_size_after = len(all_tools)
            return report

        all_tools = self.library.get_all()
        report.library_size_before = len(all_tools)

        if len(all_tools) < 2:
            report.library_size_after = len(all_tools)
            return report

        clusters = self._cluster(all_tools)
        log.info("Librarian: found %d clusters with >1 tool", len(clusters))

        for cluster in clusters:
            self._process_cluster(cluster, report)

        report.library_size_after = len(self.library.get_all())
        return report

    def _cluster(self, tools: list[dict]) -> list[list[dict]]:
        """Greedy clustering: group tools whose embeddings exceed the threshold."""
        embeddings = []
        for t in tools:
            raw = t.get("embedding")
            if raw is None:
                log.warning("Librarian: tool '%s' has no embedding, using zero vector for clustering", t["name"])
                embeddings.append(np.zeros(384, dtype=np.float32))
            else:
                embeddings.append(np.frombuffer(raw, dtype=np.float32))

        E = np.array(embeddings)
        # Normalize rows so dot product = cosine similarity
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero for zero vectors
        E = E / norms
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
                if sim >= _CLUSTER_THRESHOLD:
                    cluster.append(tools[j])
                    assigned[j] = True
            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _process_cluster(self, cluster: list[dict], report: LibrarianReport) -> None:
        """Send a cluster to the LLM, validate result, replace tools in library."""
        tools_summary = json.dumps(
            [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "source_code": t["source_code"],
                }
                for t in cluster
            ],
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
        except Exception as e:
            log.warning("Librarian LLM call failed: %s", e)
            return

        try:
            proposal = json.loads(text)
        except json.JSONDecodeError as json_err:
            log.warning("Librarian: LLM returned invalid JSON: %s", json_err)
            log.debug("Librarian raw response: %s", text[:500] if text else "")
            return

        for merge in proposal.get("merges", []):
            self._apply_merge(merge, report)

        for refactor in proposal.get("refactors", []):
            self._apply_refactor(refactor, report)

    def _apply_merge(self, merge: dict[str, Any], report: LibrarianReport) -> None:
        source = merge.get("source_code", "")
        new_name = merge.get("new_name", "")
        replace_names: list[str] = merge.get("replace_names", [])

        if not source or not new_name or not replace_names:
            return

        # Guard: proposed name must not already exist outside this cluster
        existing = self.library.get_source_code(new_name)
        if existing is not None and new_name not in replace_names:
            log.warning("Librarian: proposed merge name '%s' conflicts with existing tool, skipping", new_name)
            return

        if new_name in replace_names:
            log.warning("Librarian: merge new_name '%s' is one of the replaced names — reorder is unsafe, skipping", new_name)
            return

        # Validate: check no sandbox error (args may fail — that's OK)
        result = run_in_sandbox(
            source_code=source,
            function_name=new_name,
            args={},
            timeout=Config.VALIDATION_TIMEOUT,
        )
        if result.error:
            log.warning("Librarian: merged tool %s failed validation, skipping", new_name)
            return

        try:
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
        except Exception as e:
            log.error("Librarian: failed to apply merge %s → %s, library may have lost tools: %s", replace_names, new_name, e)
            return
        report.tools_merged += len(replace_names)
        report.details.append(f"Merged {replace_names} → {new_name}")
        log.info("Librarian: merged %s → %s", replace_names, new_name)

    def _apply_refactor(self, refactor: dict[str, Any], report: LibrarianReport) -> None:
        source = refactor.get("source_code", "")
        new_name = refactor.get("new_name", "")
        old_name = refactor.get("replace_name", "")

        if not source or not new_name or not old_name:
            return

        # Guard: proposed name must not already exist outside this rename
        existing = self.library.get_source_code(new_name)
        if existing is not None and new_name != old_name:
            log.warning("Librarian: proposed refactor name '%s' conflicts with existing tool, skipping", new_name)
            return

        result = run_in_sandbox(
            source_code=source,
            function_name=new_name,
            args={},
            timeout=Config.VALIDATION_TIMEOUT,
        )
        if result.error:
            log.warning("Librarian: refactored tool %s failed validation, skipping", new_name)
            return

        try:
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
        except Exception as e:
            log.error("Librarian: failed to apply refactor %s → %s: %s", old_name, new_name, e)
            return
        report.tools_refactored += 1
        report.details.append(f"Refactored {old_name} → {new_name}")
        log.info("Librarian: refactored %s → %s", old_name, new_name)
