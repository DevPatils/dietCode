"""Aggregate per-task benchmark runs into the table that goes in README.md.

    python3 scripts/summarize.py runs/subset

Reads each trial's results.json (the harness's verdict) alongside our own
metrics.json (steps, tokens, recovered calls), because the interesting question
is not only what passed but how the scaffold behaved on what didn't.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    try:
        # The harness writes UTF-8; some Windows tooling adds a BOM.
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    # rglob, not a fixed depth: the harness nests differently depending on how
    # --output-path was passed. Trial-level files carry a task_id; the run-level
    # summary does not, so the filter below sorts them out.
    for trial in sorted(root.rglob("results.json"), reverse=True):
        result = load(trial)
        if not isinstance(result, dict) or "task_id" not in result:
            continue
        task = result["task_id"]
        if task in seen:
            continue
        seen.add(task)

        metrics = load(trial.parent / "agent-logs" / "metrics.json") or {}
        tests = result.get("parser_results") or {}
        rows.append(
            {
                "task": task,
                "resolved": bool(result.get("is_resolved")),
                "failure_mode": result.get("failure_mode") or "",
                "agent_status": metrics.get("status", "?"),
                "steps": metrics.get("steps", 0),
                "tokens": metrics.get("total_tokens", 0),
                "tool_errors": metrics.get("tool_errors", 0),
                "recovered": metrics.get("recovered_tool_calls", 0),
                "tests_passed": sum(1 for v in tests.values() if v == "passed"),
                "tests_total": len(tests),
            }
        )
    return sorted(rows, key=lambda r: r["task"])


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/subset")
    rows = collect(root)
    if not rows:
        print(f"no results under {root}")
        return 1

    resolved = [r for r in rows if r["resolved"]]
    ran = [r for r in rows if r["agent_status"] != "?"]

    print(f"{'task':<28} {'result':<9} {'tests':<7} {'status':<22} {'steps':>5} {'tokens':>8} {'rec':>4}")
    print("-" * 92)
    for r in rows:
        tests = f"{r['tests_passed']}/{r['tests_total']}" if r["tests_total"] else "-"
        print(
            f"{r['task']:<28} {'PASS' if r['resolved'] else 'fail':<9} {tests:<7} "
            f"{r['agent_status']:<22} {r['steps']:>5} {r['tokens']:>8} {r['recovered']:>4}"
        )

    print("-" * 92)
    rate = len(resolved) / len(rows) * 100
    print(f"resolved         {len(resolved)}/{len(rows)}  ({rate:.1f}%)")
    if ran:
        print(f"avg steps        {sum(r['steps'] for r in ran) / len(ran):.1f}")
        print(f"avg tokens       {sum(r['tokens'] for r in ran) / len(ran):,.0f}")
        print(f"total tokens     {sum(r['tokens'] for r in ran):,}")
        print(f"tool errors      {sum(r['tool_errors'] for r in ran)}")
        print(f"recovered calls  {sum(r['recovered'] for r in ran)}  "
              f"(tool calls the model wrote as text)")

    modes: dict[str, int] = {}
    for r in rows:
        if not r["resolved"]:
            modes[r["agent_status"]] = modes.get(r["agent_status"], 0) + 1
    if modes:
        print("\nhow the agent ended on failures:")
        for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
            print(f"  {mode:<24} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
