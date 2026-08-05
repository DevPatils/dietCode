# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CLI coding agent: an agentic loop with tool-calling that reads/writes files and runs
shell commands in a Docker sandbox until a task is done, benchmarked against
Terminal-Bench. [plannings.md](plannings.md) is the design document and remains the plan
of record; [README.md](README.md) is where benchmark results go.

## Commands

```bash
pip install -e ".[dev]"                    # editable install; provides the `dietcode` command
python -m pytest                           # full suite; Docker tests skip if the daemon is down
python -m pytest tests/test_loop.py        # one file
python -m pytest -k "timeout"              # one test by name
dietcode "task description"                # runs in the current directory, asking first
dietcode --sandbox "task description"      # runs in a container instead
python cli.py "task description"           # same code, from a checkout
```

**`cli.py` at the repo root is a two-line shim.** The real entrypoint is
[agent/cli.py](agent/cli.py), so the installed `dietcode` command and the checkout path
execute identical code. Console script is `dietcode = "agent.cli:entrypoint"`.

**Credentials never live in the project.** [agent/auth.py](agent/auth.py) stores keys in
the OS keychain, falling back to `~/.dietcode/credentials.json` at 0600. Resolution
order is env var, then saved login — the env var wins so CI can override without
touching a user's login. Keys are cleaned of BOMs and zero-width characters on the way
in: `str.strip()` leaves them, and PowerShell adds a BOM when piping, which silently
corrupts the stored key and surfaces later as an unexplained 401.

**Two interpreters.** The agent runs on Python 3.11 (the default `python` here).
terminal-bench requires ≥3.12 and is installed on 3.13. So:

```bash
py -3.13 -m pytest tests/test_adapter.py   # adapter tests SKIP on 3.11 -- run them here
```

**The benchmark harness does not run on Windows** — it builds container paths with
`pathlib`, so `/tmp` becomes `\tmp` and it dies in `TmuxSession.__init__` before the
agent is called. Run it from WSL via the wrapper, which encodes four separate
0.2.18 workarounds (see the README's benchmark section):

```bash
wsl bash scripts/benchmark.sh                 # hello-world
wsl bash scripts/benchmark.sh broken-python   # one task
```

A harness `0.00%` is ambiguous: check `total_input_tokens` in `results.json`. `null`
means the harness failed before the agent ran, not that the agent failed.

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

Everything that reads a keypress lives in `agent/prompts.py` (pickers, confirm, secret
entry) and `agent/completion.py` (slash completion). `agent/models.py` asks a provider
what it will actually accept, filtering `/models` down to ids that can drive a
tool-calling loop — an embedding id in the picker is a 400 waiting to happen.

**Rendering stays in `agent/ui.py`.** The loop emits events and never prints, so the
benchmark adapter runs it with no console attached. Anything that makes `agent_loop`
aware of a terminal breaks that.

**Streaming is opt-in (`stream=True`), and the benchmark deliberately leaves it off.**
Streaming requires reassembling tool calls from fragments — more machinery to go wrong,
for output no scored run watches. Both transports normalize to `Completion(content,
tool_calls, usage)` in `_from_response` / `_from_stream`, so the loop body is transport-
agnostic. When adding a field to a model turn, add it to `Completion` and to *both*
constructors, or streamed runs silently lose it.

Reassembly gotchas, each covered by a test in `tests/test_streaming.py`: tool-call
fragments are keyed by `index` (several calls stream interleaved); the function name is
accumulated but an immediate repeat is skipped, since some providers send the name whole
in every delta and others chunk it; usage arrives in a final chunk with **no choices**,
so a `continue` on empty choices before reading usage blanks the token column;
`stream_options={"include_usage": True}` is not universally supported and falls back.
When streaming, the loop emits `assistant_delta` and suppresses `assistant_text` —
emitting both double-prints the reply.

The load-bearing idea: **the CLI and the benchmark differ only in which `Executor` is
passed to `agent_loop`.** [adapters/terminal_bench.py](adapters/terminal_bench.py)'s
`SessionExecutor` subclasses `DockerExecutor` and overrides exactly one method,
`_raw_exec`. Everything else — tool logic, cwd persistence, file encoding — is inherited,
so the benchmark exercises the same code the CLI does. Adding a second implementation of
any tool behaviour in the adapter defeats the entire design.

`agent_loop` returns an `AgentResult` with status `complete` | `stopped` |
`max_iterations_reached` | `budget_exhausted` | `error`, plus steps, tool-call counts
and token usage — `.metrics()` is what feeds the README table.

**Seven tools.** `read_file`, `write_file`, `edit_file`, `find_files`, `search`,
`run_shell`, `task_complete`. `spawn_subagent` is an eighth, opt-in via `--subagents`.

- **`edit_file` never guesses.** A snippet matches exactly once or the call fails with
  a reason the model can act on — including a specific "the text is there but the
  whitespace differs" hint, which is the usual cause. Editing the wrong place silently
  is worse than not editing.
- **`find_files` / `search` live on the `Executor`, not in tools.py.** The two backends
  do them completely differently: the container has `find` and `grep`, the host may be
  Windows and have neither. A shell-based implementation in tools.py broke `--here` on
  Windows — that is why the capability moved down a layer.
- **`spawn_subagent` returns only the child's summary.** The isolation is the mechanism
  being tested; passing the transcript back would grow the parent's context exactly as
  fast as doing the work inline. Depth is capped at 1 — a child that can spawn children
  recurses until the daily request quota is gone.

**Project instructions** (`DIETCODE.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`) are
read from the *host*, never through the `Executor`, so the agent cannot rewrite its own
standing orders mid-run.

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
  container mid-benchmark fails the task. `close()` also only runs on a *clean* exit,
  which is why every container carries a `com.dietcode.agent` label and startup calls
  `sweep_orphans()`. The sweep is age-based (6h) on purpose: there is no reliable way to
  tell whether the owning process is alive, and killing a container out from under a
  running session is worse than leaving a stale one.
- **Trimming must never split a tool_call from its tool result.** `trim_messages` groups
  messages into atomic blocks via `_blocks()` for exactly this reason — half a pair makes
  the API reject the entire request. It also drops any leading orphan `tool` message.
  Trimming applies to what is *sent*; `result.messages` keeps the full transcript.
- **On `context_length_exceeded`, shrink relative to what was actually sent**, not to the
  configured budget. Halving a 48k budget when the real payload is 2k changes nothing and
  the retry sends an identical request — that exact bug was caught by a test.
- **Classify retryable errors by type and status, not message text** (`_is_transient`).
  String-matching an error body is how the `tool_use_failed` bug hid; a typed 4xx must
  never be retried.
- **Never print a non-ASCII character without `glyph()`.** Windows consoles default to
  cp1252; printing `→` raises `UnicodeEncodeError` mid-render and killed the first
  interactive session outright. `cli.py` calls `use_utf8_stdout()` before anything
  prints, and `ui.glyph()` falls back to ASCII when the encoding is narrow. The same
  applies to rich's box-drawing: `banner()` picks `box.ASCII` from `ascii_only()`
  rather than trusting rich's terminal detection.
- **Tool schemas are narrowed per provider on the way out** (`tools_for`). The union
  types (`["integer","string"]`) exist because Groq validates the model's arguments
  server-side; Gemini's schema layer is OpenAPI-derived and cannot express a union at
  all. So `TOOLS` stays canonical and each provider gets its own shape. `_rebuild_client`
  re-narrows on `/provider`, or the next turn 400s on a schema built for the provider
  just left.
- **Anything the user types at a prompt must be echoed by prompt_toolkit, not
  `input()`.** `input()` writes straight to the terminal while rich is still repainting
  a spinner over the same line, so the keystrokes vanish and the user answers the
  permission gate blind. `ui._read_answer` and `prompts.confirm` both own the line.
- **Enter approves only what is safe to approve by reflex.** The permission prompt
  defaults to yes, because approving reads is most of what it does — but not for
  `Risk.DANGEROUS` or anything `outside_root`.
- **A picker with no terminal cancels; it never guesses.** `prompts.choose` returns
  `None` when stdin is not a tty. Returning the first option would silently change a
  provider or model in a scripted run.
- **The suite makes no network calls, and `tests/conftest.py` enforces it.** A test that
  called `main()` loaded the developer's `.env`, found a real key and spent tokens on a
  live request. httpx is blocked; the Docker tests use requests and still work.
- **`load_dotenv` must be given `find_dotenv(usecwd=True)`.** The default searches
  upward from the calling *file*, which inside a pipx install is site-packages — so an
  installed `dietcode` never saw the `.env` in the directory the user was standing in.

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
