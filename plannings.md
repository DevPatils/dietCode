# Project: CLI Coding Agent (Weekend Build)

## Goal
Build a CLI agent that can read/write files, run shell commands, and loop until
task completion (agentic loop with tool-calling). Then benchmark it against
Terminal-Bench 2.0. Stretch goal: add sub-agent orchestration and see if the
benchmark score improves.

**Success criteria:** working agent loop + tool-calling, a real benchmark score
(not just "it seems to work"), and a documented before/after if sub-agents are added.

---

## Tech Stack

- **Language:** Python 3.11+
- **LLM Provider:** Groq API (free tier) — OpenAI-compatible endpoint at
  `https://api.groq.com/openai/v1`
- **Model:** `llama-3.3-70b-versatile` (fallback: `qwen/qwen3-32b` if tool-calling
  reliability is worse on the primary model)
- **No agent framework** — raw API calls + a hand-rolled loop. Do NOT use LangChain
  or similar. The point is to implement and understand the loop directly.
- **Sandboxing:** Docker — all shell command execution happens inside a container,
  from the very first version, so local dev and the benchmark harness share the
  same code path.
- **Benchmark:** `terminal-bench` (pip package, Harbor harness)

---

## Project Structure

```
cli-agent/
├── agent/
│   ├── loop.py             # the agent loop
│   ├── tools.py             # tool schemas + execute_tool dispatch
│   └── sandbox.py           # docker exec wrapper for run_shell
├── adapters/
│   └── terminal_bench.py    # Harbor-compatible adapter
├── cli.py                   # entrypoint: python cli.py "task description"
├── requirements.txt
└── README.md                 # benchmark results go here
```

---

## Tools to Implement

1. **`read_file`** — read contents of a file
   - input: `{ "path": string }`
2. **`write_file`** — write/overwrite a file
   - input: `{ "path": string, "content": string }`
3. **`run_shell`** — execute a shell command inside the sandboxed Docker container,
   return stdout, stderr, and exit code
   - input: `{ "command": string, "timeout": integer (default 30s) }`
4. **`task_complete`** — explicit termination signal, called when the agent
   believes the task is fully done
   - input: `{ "summary": string }`

Note: free/open models (Llama, Qwen) are less reliable at producing well-formed
tool calls than closed frontier models. The tool dispatcher MUST handle malformed
tool-call JSON or invalid tool names gracefully — return a clear error string as
the tool result instead of crashing the loop. Treat this defensively from the start,
not as an afterthought.

---

## Agent Loop (core logic)

Pseudocode:

```
messages = [user: task_prompt]

for i in range(max_iterations):
    response = call_llm(messages, tools=TOOLS)
    messages.append(assistant: response)

    tool_calls = extract_tool_calls(response)
    if no tool_calls:
        break  # model stopped without acting — treat as done or stuck

    tool_results = []
    for call in tool_calls:
        if call.name == "task_complete":
            return { status: "complete", summary: call.input.summary, steps: i+1 }
        result = execute_tool(call.name, call.input)  # must not throw
        tool_results.append(tool_result for call.id)

    messages.append(user: tool_results)

return { status: "max_iterations_reached", steps: max_iterations }
```

Keep `max_iterations` tight (10–15) during development to avoid burning Groq's
free-tier daily request quota (~1,000 requests/day) on runaway loops.

---

## Benchmark Integration (Terminal-Bench 2.0)

- Install: `pip install terminal-bench` (or `uv tool install terminal-bench`) —
  requires Docker running locally.
- Terminal-Bench 2.0 has 89 hand-verified tasks across categories (sysadmin,
  security, ML, data science, software engineering, etc.), each run inside an
  isolated Docker container and graded by checking the final container state
  against automated tests — not by reading the transcript.
- Write a thin **Harbor-compatible adapter** in `adapters/terminal_bench.py` that
  wraps the same `agent_loop` from `agent/loop.py`, but routes `run_shell` to
  execute inside the task's own container (provided by the harness) instead of a
  locally-managed one.
- Don't run all 89 tasks repeatedly during development — use a fixed subset of
  ~15–20 tasks across a few categories for fast iteration.
- Reference baseline: **Terminus** is the benchmark's own minimal reference agent
  (LLM + terminal, no extra scaffolding). If time allows, run it on the same
  subset for a "how much of my score comes from the model vs. my scaffold" comparison.

### Metrics to capture and report in README.md
- Resolution rate (% tasks passed)
- Steps/iterations per task
- Tokens used per task
- Failure mode breakdown (infinite loop, misread tool output, malformed tool call,
  gave up early, wrong file path, etc.)

---

## Stretch Goal: Sub-Agent Orchestration

Add a `spawn_subagent(task_description)` tool to the orchestrator's tool list.
When called:
1. Start a **fresh** `agent_loop` with its own isolated message history (same
   tool set as the parent).
2. Run it to completion (or max iterations).
3. Return only the **final summary** to the parent — not the full sub-agent
   transcript. This context isolation (not shared memory) is the actual
   mechanism Claude Code uses for its own sub-agent/Task tool.

Test this specifically on multi-file tasks (e.g., a multi-file refactor) and
compare resolution rate / step-count against the non-orchestrated baseline on
the same tasks. Report the before/after numbers.

---

## Build Order (suggested)

1. `agent/tools.py` — tool schemas + `execute_tool` dispatch (local filesystem
   + subprocess first, no Docker yet)
2. `agent/loop.py` — the loop, tested on toy tasks against Groq directly
3. `agent/sandbox.py` — move `run_shell` execution into Docker
4. `adapters/terminal_bench.py` — wire the loop into the Harbor interface
5. Run benchmark subset, capture metrics, iterate on failure modes
6. (Stretch) `spawn_subagent` tool + orchestrator changes, re-run benchmark
7. Write up results in README.md

---

## Explicit Non-Goals (for this weekend)
- No LangChain or other agent framework
- No fancy diff-based file editing (full-file overwrite is fine for v1)
- No UI beyond the CLI
- No running the full 89-task Terminal-Bench suite repeatedly — fixed subset only