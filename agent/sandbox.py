from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class SandboxResult:
    success: bool
    result: str | None = None
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


def run_in_sandbox(
    source_code: str,
    function_name: str,
    args: dict,
    timeout: int = 10,
) -> SandboxResult:
    runner = _build_runner(source_code, function_name, args)

    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="toolforge_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(runner)

        from config import Config
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Config.WORKSPACE_DIR,
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0 and not stdout:
            return SandboxResult(
                success=False,
                error=stderr or f"Process exited with code {proc.returncode}",
                stdout=stdout,
                stderr=stderr,
            )

        # Parse the last line of stdout as JSON result
        lines = stdout.split("\n")
        result_line = ""
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                result_line = line
                break

        if not result_line:
            return SandboxResult(
                success=False,
                error=f"No JSON result in output. stdout={stdout[:500]}, stderr={stderr[:500]}",
                stdout=stdout,
                stderr=stderr,
            )

        output = json.loads(result_line)
        if output.get("status") == "ok":
            result_val = output.get("result")
            if not isinstance(result_val, str):
                result_val = json.dumps(result_val)
            return SandboxResult(success=True, result=result_val, stdout=stdout, stderr=stderr)
        else:
            return SandboxResult(
                success=False,
                error=output.get("error", "Unknown error"),
                stdout=stdout,
                stderr=stderr,
            )

    except subprocess.TimeoutExpired:
        return SandboxResult(success=False, error=f"Execution timed out after {timeout}s")
    except json.JSONDecodeError as e:
        return SandboxResult(success=False, error=f"Failed to parse output JSON: {e}")
    except Exception as e:
        return SandboxResult(success=False, error=f"Sandbox error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _build_runner(source_code: str, function_name: str, args: dict) -> str:
    args_json = json.dumps(args)
    return f'''import json
import sys

# --- Generated tool code ---
{source_code}

# --- Invoke ---
try:
    _args = json.loads({json.dumps(args_json)})
    _result = {function_name}(**_args)
    print(json.dumps({{"status": "ok", "result": _result}}))
except Exception as _e:
    print(json.dumps({{"status": "error", "error": str(_e)}}))
'''
