from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from agent.prompts import (
    TOOL_GENERATOR_SYSTEM_PROMPT,
    build_tool_fix_prompt,
    build_tool_gen_user_prompt,
)
from agent.sandbox import SandboxResult, run_in_sandbox
from config import Config

log = logging.getLogger(__name__)


class ToolGenerator:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def generate(
        self,
        capability_needed: str,
        capability_detail: str,
        task_context: str,
    ) -> dict[str, Any] | None:
        user_prompt, function_name = build_tool_gen_user_prompt(
            capability_needed, capability_detail, task_context
        )

        spec = self._call_claude(TOOL_GENERATOR_SYSTEM_PROMPT, user_prompt)
        if spec is None:
            return None

        # Override function_name to ensure consistency
        spec["function_name"] = function_name
        spec["name"] = function_name

        # Validate via sandbox
        for attempt in range(Config.TOOL_GEN_RETRIES):
            validation = self._validate(spec)
            if validation.success:
                log.info("Tool %s validated on attempt %d", function_name, attempt + 1)
                return spec

            log.warning(
                "Tool %s failed validation attempt %d: %s",
                function_name, attempt + 1, validation.error,
            )

            if attempt < Config.TOOL_GEN_RETRIES - 1:
                fix_prompt = build_tool_fix_prompt(
                    function_name=function_name,
                    previous_code=spec.get("source_code", ""),
                    error=validation.error or "Unknown error",
                    test_call=spec.get("test_call", {}),
                )
                fixed = self._call_claude(TOOL_GENERATOR_SYSTEM_PROMPT, fix_prompt)
                if fixed:
                    fixed["function_name"] = function_name
                    fixed["name"] = function_name
                    spec = fixed

        log.error("Tool %s failed all %d validation attempts", function_name, Config.TOOL_GEN_RETRIES)
        return None

    def _validate(self, spec: dict[str, Any]) -> SandboxResult:
        source = spec.get("source_code", "")
        name = spec.get("function_name", "")
        test_call = spec.get("test_call", {})

        if not source or not name:
            return SandboxResult(success=False, error="Missing source_code or function_name")

        if not isinstance(test_call, dict):
            return SandboxResult(success=False, error="test_call must be a dict of args")

        return run_in_sandbox(
            source_code=source,
            function_name=name,
            args=test_call,
            timeout=Config.VALIDATION_TIMEOUT,
        )

    def _call_claude(self, system: str, user: str) -> dict[str, Any] | None:
        try:
            response = self.client.messages.create(
                model=Config.MODEL,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )

            text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            return json.loads(text)

        except json.JSONDecodeError as e:
            log.error("Failed to parse tool generator response as JSON: %s", e)
            return None
        except Exception as e:
            log.error("Claude API error in tool generator: %s", e)
            return None
