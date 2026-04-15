from __future__ import annotations

import json
import logging
import queue
from typing import Any

import anthropic

from agent.failure_store import FailureStore
from agent.prompts import build_agent_system_prompt
from agent.retriever import embed, ToolRetriever
from agent.sandbox import run_in_sandbox
from agent.tool_generator import ToolGenerator
from config import Config
from library.tool_library import ToolLibrary

log = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        tool_library: ToolLibrary,
        event_queue: queue.Queue | None = None,
    ) -> None:
        self.library = tool_library
        self.events = event_queue
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.generator = ToolGenerator()
        self.failure_store = FailureStore(db_path=Config.FAILURES_PATH)
        self.retriever = ToolRetriever(library=tool_library, threshold=Config.RETRIEVAL_THRESHOLD)
        self.messages: list[dict[str, str]] = []
        self.iteration = 0
        self._tool_created: bool = False
        self._tool_reused: bool = False
        self._tool_name_used: str = ""

    def run(self, task: str) -> dict[str, Any]:
        self.messages = [{"role": "user", "content": task}]
        self.iteration = 0
        self._tool_created = False
        self._tool_reused = False
        self._tool_name_used = ""

        self._emit("task_start", {"task": task})

        while self.iteration < Config.MAX_ITERATIONS:
            self.iteration += 1

            response = self._call_claude(task)
            if response is None:
                self._emit("error", {"content": "Failed to get a response from Claude"})
                return {"success": False, "error": "Claude API failure", "tool_created": False, "tool_reused": False, "tool_used": "", "attempts": self.iteration}

            thought = response.get("thought", "")
            action = response.get("action", "")

            self._emit("thought", {"content": thought, "iteration": self.iteration})

            if action == "call_tool":
                result = self._handle_call_tool(response)
                self.messages.append({"role": "assistant", "content": json.dumps(response)})
                self.messages.append({"role": "user", "content": f"Tool result:\n{result}"})

            elif action == "acquire_tool":
                acquired = self._handle_acquire_tool(response, task)
                self.messages.append({"role": "assistant", "content": json.dumps(response)})
                if acquired:
                    self.messages.append({
                        "role": "user",
                        "content": f"New tool acquired: {acquired}. You can now use it by calling it with action=call_tool.",
                    })
                else:
                    self.messages.append({
                        "role": "user",
                        "content": "Tool acquisition failed. Try a different approach or use existing tools.",
                    })

            elif action == "final_answer":
                answer = response.get("answer", "")
                self._emit("final_answer", {"content": answer})
                return {
                    "success": True,
                    "answer": answer,
                    "tool_created": self._tool_created,
                    "tool_reused": self._tool_reused,
                    "tool_used": self._tool_name_used,
                    "attempts": self.iteration,
                }

            elif action == "impossible":
                reason = response.get("reason", "Unknown reason")
                self._emit("impossible", {"content": reason})
                return {
                    "success": False,
                    "reason": reason,
                    "tool_created": self._tool_created,
                    "tool_reused": self._tool_reused,
                    "tool_used": self._tool_name_used,
                    "attempts": self.iteration,
                }

            else:
                self._emit("error", {"content": f"Unknown action: {action}"})
                self.messages.append({"role": "assistant", "content": json.dumps(response)})
                self.messages.append({
                    "role": "user",
                    "content": f"Invalid action '{action}'. Use one of: call_tool, acquire_tool, final_answer, impossible.",
                })

        self._emit("error", {"content": "Max iterations reached"})
        return {"success": False, "error": "Max iterations reached", "tool_created": self._tool_created, "tool_reused": self._tool_reused, "tool_used": self._tool_name_used, "attempts": self.iteration}

    def _call_claude(self, task: str) -> dict[str, Any] | None:
        if Config.ABLATION_NO_LIBRARY:
            tools_for_prompt = []
        else:
            relevant = self.retriever.retrieve(task, top_k=Config.RETRIEVAL_TOP_K)
            if relevant:
                tools_for_prompt = [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "args": t["args"],
                        "returns": t["returns"],
                    }
                    for t in relevant
                ]
            else:
                tools_for_prompt = []
        system = build_agent_system_prompt(tools_for_prompt)

        try:
            response = self.client.messages.create(
                model=Config.MODEL,
                max_tokens=4096,
                system=system,
                messages=self.messages,
            )

            text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            # Extract first complete JSON object (handles trailing text after closing brace)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text)
                return obj

        except json.JSONDecodeError as e:
            log.error("Failed to parse agent response as JSON: %s\nRaw: %s", e, text[:500] if 'text' in dir() else "N/A")
            return None
        except Exception as e:
            log.error("Claude API error in agent loop: %s", e)
            return None

    def _handle_call_tool(self, response: dict[str, Any]) -> str:
        tool_name = response.get("tool_name", "")
        tool_args = response.get("tool_args", {})

        self._emit("tool_call", {"tool": tool_name, "args": tool_args})

        source = self.library.get_source_code(tool_name)
        if source is None:
            error = f"Tool '{tool_name}' not found in library"
            self._emit("tool_result", {"content": error, "success": False})
            return error

        result = run_in_sandbox(
            source_code=source,
            function_name=tool_name,
            args=tool_args,
            timeout=Config.SANDBOX_TIMEOUT,
        )

        if result.success:
            self.library.increment_use(tool_name)
            self._tool_reused = True
            self._tool_name_used = tool_name
            self._emit("tool_result", {"content": result.result or "", "success": True})
            return result.result or ""
        else:
            error_msg = f"Tool execution error: {result.error}"
            self._emit("tool_result", {"content": error_msg, "success": False})
            return error_msg

    def _handle_acquire_tool(self, response: dict[str, Any], task: str) -> str | None:
        capability_needed = response.get("capability_needed", "")
        capability_detail = response.get("capability_detail", "")

        self._emit("acquiring_tool", {
            "capability": capability_needed,
            "detail": capability_detail,
        })

        failures = self.failure_store.get_recent(task, limit=3)
        spec = self.generator.generate(
            capability_needed=capability_needed,
            capability_detail=capability_detail,
            task_context=task,
            failures=failures,
        )

        if spec is None:
            self.failure_store.log(task, "<generation failed>", "LLM refused or produced invalid Python")
            self._emit("tool_acquisition_failed", {"capability": capability_needed})
            return None

        self.library.add_tool(spec, embedding=embed(spec.get("description", spec["name"])), task_context=task)
        self.failure_store.clear(task)
        self._tool_created = True
        self._tool_name_used = spec["name"]

        self._emit("tool_acquired", {
            "tool_name": spec["name"],
            "description": spec.get("description", ""),
            "tags": spec.get("tags", []),
        })

        return spec["name"]

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, **data}
        log.info("Event: %s", json.dumps(event, default=str)[:200])
        if self.events:
            self.events.put(event)
