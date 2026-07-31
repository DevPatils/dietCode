# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CLI coding agent: an agentic loop with tool-calling that reads/writes files and runs
shell commands in a Docker sandbox until a task is done, benchmarked against
Terminal-Bench. [plannings.md](plannings.md) is the design document and remains the plan
of record; [README.md](README.md) is where benchmark results go.

## Commands

```bash
python -m pytest                          # full suite; Docker tests skip if the daemon is down
python -m pytest tests/test_loop.py        # one file
python -m pytest -k "timeout"              # one test by name
python cli.py "task description"           # run the agent (needs GROQ_API_KEY + Docker)
python cli.py --local --workdir /tmp/x "…"  # host execution, no isolation, dev only
```

**Two interpreters.** The agent runs on Python 3.11 (the default `python` here).
terminal-bench requires ≥3.12 and is installed on 3.13. So:

```bash
py -3.13 -m pytest tests/test_adapter.py   # adapter tests SKIP on 3.11 -- run them here
py -3.13 -m pip install terminal-bench
tb run --dataset terminal-bench-core --agent-import-path adapters.terminal_bench:CliAgent \
       --model llama-3.3-70b-versatile --task-id hello-world
```

A green `python -m pytest` does **not** mean the adapter is tested — it skips silently
on 3.11. Run the 3.13 command too before claiming the adapter works.

The loop tests use a scripted fake client ([tests/fake_llm.py](tests/fake_llm.py)), so
the suite needs no API key and makes no network calls.

## Architecture

```
interactive ─┐   (agent/repl.py + agent/ui.py)
one-shot   ──┼─> agent_loop ──> execute_tool ──> Executor ──> container
tb run     ──┘   agent/loop.py   agent/tools.py   agent/sandbox.py
```

`python cli.py` with no task argument enters the interactive session; with a task
it runs once and exits. Interactive mode reuses one container and feeds the previous
turn's `.messages` back in as `history` — that is the *only* behavioural difference.
`history` is copied, not mutated, so an interrupted turn cannot leave the caller with
a transcript containing unanswered tool calls (which the API rejects).

**Rendering stays in `agent/ui.py`.** The loop emits events and never prints, so the
benchmark adapter runs it with no console attached. Anything that makes `agent_loop`
aware of a terminal breaks that.

The load-bearing idea: **the CLI and the benchmark differ only in which `Executor` is
passed to `agent_loop`.** [adapters/terminal_bench.py](adapters/terminal_bench.py)'s
`SessionExecutor` subclasses `DockerExecutor` and overrides exactly one method,
`_raw_exec`. Everything else — tool logic, cwd persistence, file encoding — is inherited,
so the benchmark exercises the same code the CLI does. Adding a second implementation of
any tool behaviour in the adapter defeats the entire design.

`agent_loop` returns an `AgentResult` with status `complete` | `stopped` |
`max_iterations_reached` | `error`, plus steps, tool-call counts and token usage —
`.metrics()` is what feeds the README table.

## Invariants

These encode failures already hit; changing them will silently break runs.

- **`execute_tool` never raises.** Llama/Qwen emit malformed tool-call JSON, invented
  tool names and wrong-typed arguments routinely. Every failure returns an error string
  the model can read and correct. `tests/test_tools.py` asserts this for each case,
  including an executor whose every method throws.
- **A tool call written as message text still has to run.** Observed on the first real
  run: llama-3.3-70b emitted `<function/run_shell {...}</function>` as content instead of
  using the tool-calling API, so the loop saw no tool calls and stopped at step 1.
  `extract_tool_calls_from_text` recovers the known formats and the loop rewrites the
  call into the transcript structurally (clearing the malformed text, so history shows
  the model a correct example). It must never invent a call from prose that merely
  mentions a tool — recovery is gated on the name matching a real tool and the arguments
  decoding as JSON. Tracked as `recovered_tool_calls`, separate from `tool_errors`,
  because it measures the model rather than the scaffold.
- **`task_complete` batched with the work is deferred, not honoured.** Models routinely
  emit the whole task — write, run, *and* `task_complete` — in one turn, declaring the
  output verified before any tool result existed. Observed live: a run wrote bash into a
  `.py` file, got a `SyntaxError`, and claimed success in the same breath. The loop now
  runs every call in the batch, hands back the results, and requires `task_complete` on
  its own turn. Bounded by `MAX_COMPLETION_DEFERRALS` so a model that always batches
  still terminates. Never `return` mid-batch: it skips later calls and leaves their
  `tool_call_id`s unanswered, corrupting the transcript.
- **Tool schemas must be permissive where the dispatcher coerces.** Groq validates tool
  arguments against `TOOLS` server-side and rejects the entire generation with a 400 —
  a model sending `"timeout": "10"` killed a run, even though `_coerce_timeout` handles
  it fine. Hence `timeout` accepts `["integer", "string"]`. When it happens anyway, the
  400 body carries `failed_generation`, and `failed_generation_text` salvages the call
  from it rather than losing the task. Only `code == "tool_use_failed"` is recovered;
  other 400s (bad key, etc.) must still surface as errors.
- **File tools go through the `Executor`, never the host filesystem.** Otherwise the
  benchmark agent reads the host while its shell acts in the container.
- **Every `tool_call` must get a `tool` message with a matching `tool_call_id`**, or the
  next API request is rejected. The loop synthesizes ids when the model omits them, and
  answers `task_complete` before returning so the transcript stays valid.
- **The shell wrapper persists the working directory across calls** via
  `/tmp/.agent_cwd`. Each `docker exec` is a fresh process, so `cd /app` would otherwise
  be lost by the next command and the agent would act blind. The `pwd` capture must
  happen *in the same shell that ran the command* — capturing it in the outer wrapper
  records the wrapper's directory and loses every `cd`.
- **The agent's command reaches the shell as an argv element, never interpolated** into
  the wrapper script; file content is base64'd over argv. Nothing the model generates can
  be reparsed as shell syntax. Cost: writes are capped by `ARG_MAX` (~1MB).
- **`close()` only removes containers we created.** Tearing down a harness-owned
  container mid-benchmark fails the task.
- **Never print a non-ASCII character without `glyph()`.** Windows consoles default to
  cp1252; printing `→` raises `UnicodeEncodeError` mid-render and killed the first
  interactive session outright. `cli.py` calls `use_utf8_stdout()` before anything
  prints, and `ui.glyph()` falls back to ASCII when the encoding is narrow. The same
  applies to rich's box-drawing: `banner()` picks `box.ASCII` from `ascii_only()`
  rather than trusting rich's terminal detection.

## Constraints from the plan

- **No agent framework** — raw OpenAI-compatible calls against Groq
  (`https://api.groq.com/openai/v1`), model `llama-3.3-70b-versatile`, fallback
  `qwen/qwen3-32b` only if tool-calling is worse on the primary.
- **Keep `max_iterations` at 10–15.** Groq's free tier is ~1,000 requests/day and each
  loop step is one request.
- **Never run the full 89-task suite while iterating** — fixed ~15–20 task subset.
- Full-file overwrite is fine for `write_file`; no diff-based editing, no UI beyond CLI.

## Not built yet

`spawn_subagent` (stretch goal) — a fresh `agent_loop` with isolated message history that
returns *only* its final summary to the parent; the context isolation is the mechanism
being tested, so sharing history defeats it. `agent_loop` already takes an
`extra_tool_handlers` hook for this and the hook is tested. It is deliberately left until
a baseline benchmark score exists to compare against.

Terminal-Bench grades by inspecting final container state, not the agent transcript — a
run that looks correct in the log can still fail. Note also that `SessionExecutor`
bypasses tmux, so agent commands do not appear in a task's asciinema recording; use the
`transcript.json` written to the harness logging dir for failure analysis.
