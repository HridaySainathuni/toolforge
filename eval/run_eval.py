"""
Evaluation harness for ToolForge.

Usage:
    python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition full
    python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition no_library
    python -m eval.run_eval --batch eval/tasks/math_batch1.jsonl --condition no_abstraction
    python -m eval.run_eval --batch eval/tasks/text_batch2.jsonl --condition full
    python -m eval.run_eval --batch eval/tasks/mixed_batch3.jsonl --condition full

Conditions:
    full           - Full system (library + abstraction prompt)
    no_library     - No tool library (fresh LLM call, no tool reuse)
    no_abstraction - Library but no abstraction prompt in tool creator
    baseline       - Alias for no_library

Results written to eval/results/<batch>_<condition>_<timestamp>.csv
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
    """Loose comparison: strip whitespace, try numeric equality, fall back to string."""
    import re
    a = actual.strip().lower()
    e = expected.strip().lower()
    if a == e:
        return True
    # Try numeric comparison
    try:
        return abs(float(a) - float(e)) < 1e-3
    except ValueError:
        pass
    # Try extracting first number from actual answer
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
    # Apply ablation flags via environment variables
    if condition in ("no_library", "baseline"):
        os.environ["ABLATION_NO_LIBRARY"] = "true"
        os.environ.pop("ABLATION_NO_ABSTRACTION", None)
    elif condition == "no_abstraction":
        os.environ["ABLATION_NO_ABSTRACTION"] = "true"
        os.environ.pop("ABLATION_NO_LIBRARY", None)
    else:  # full
        os.environ.pop("ABLATION_NO_LIBRARY", None)
        os.environ.pop("ABLATION_NO_ABSTRACTION", None)

    # Reload config flags (they're read at class definition time — reimport)
    import importlib
    import config as cfg_module
    importlib.reload(cfg_module)
    # Patch Config class attributes directly
    from config import Config as Cfg
    Cfg.ABLATION_NO_LIBRARY = os.getenv("ABLATION_NO_LIBRARY", "false").lower() == "true"
    Cfg.ABLATION_NO_ABSTRACTION = os.getenv("ABLATION_NO_ABSTRACTION", "false").lower() == "true"

    tasks = load_tasks(batch_path)
    batch_name = os.path.splitext(os.path.basename(batch_path))[0]

    os.makedirs("eval/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"eval/results/{batch_name}_{condition}_{timestamp}.csv"

    # Fresh library per condition run (so ablations start from same baseline)
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
            log.info("[%d/%d] %s: %s", i + 1, len(tasks), t["id"], t["task"][:70])
            start = time.time()

            try:
                loop = AgentLoop(tool_library=library)
                result = loop.run(t["task"])
                elapsed = time.time() - start

                actual = str(result.get("answer", "") or "").strip()
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
                    "attempts": result.get("attempts", 0),
                    "time_seconds": round(elapsed, 2),
                    "condition": condition,
                })
                csvfile.flush()

            except Exception as e:
                elapsed = time.time() - start
                log.error("Task %s raised exception: %s", t["id"], e)
                writer.writerow({
                    "task_id": t["id"],
                    "domain": t.get("domain", ""),
                    "task": t["task"],
                    "expected": t["expected"],
                    "actual": "",
                    "success": False,
                    "tool_created": False,
                    "tool_reused": False,
                    "tool_name": "",
                    "attempts": 0,
                    "time_seconds": round(elapsed, 2),
                    "condition": condition,
                })
                csvfile.flush()

    log.info("Results written to %s", out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToolForge evaluation harness")
    parser.add_argument("--batch", required=True, help="Path to .jsonl task file")
    parser.add_argument(
        "--condition",
        choices=["full", "no_library", "no_abstraction", "baseline"],
        default="full",
        help="Ablation condition to run",
    )
    args = parser.parse_args()
    run_eval(args.batch, args.condition)
