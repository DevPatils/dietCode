# CLI Coding Agent

A command-line coding agent: an agentic loop with tool-calling that reads/writes
files and runs shell commands in a Docker sandbox until a task is done. No agent
framework — raw OpenAI-compatible API calls against Groq and a hand-rolled loop.

Benchmarked against Terminal-Bench.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your Groq key in it
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).
Docker must be running.

## Usage

```bash
python cli.py "write a python script that prints the first 20 primes and run it"
```

By default the agent works in a throwaway container, so **anything it writes is
discarded when the run ends**. To keep the files, mount a directory — the agent
stays sandboxed, but `/workspace` is now a real folder on your machine:

```bash
python cli.py --mount ./my-project "add a test for the parser and make it pass"
```

| Flag | Meaning |
| --- | --- |
| `--mount HOSTDIR[:TARGET]` | bind-mount a host directory into the sandbox so the agent's files persist (default target `/workspace`). Repeatable |
| `--local` | run on the host instead of Docker (no isolation — dev only) |
| `--container NAME` | attach to an existing container instead of creating one |
| `--image IMAGE` | sandbox image (default `python:3.11-slim`) |
| `--model NAME` | default `llama-3.3-70b-versatile` |
| `--max-iterations N` | default 12 |
| `--json` | print metrics as JSON |
| `--quiet` | only print the final result |

Exit code is 0 when the agent called `task_complete`, 1 otherwise.

## Tests

```bash
python -m pytest                       # Docker tests skip if the daemon is down
python -m pytest tests/test_loop.py     # one file
python -m pytest -k timeout             # one test
```

The loop tests use a scripted fake client (`tests/fake_llm.py`), so the suite
needs no API key and makes no network calls.

## How it works

```
cli.py ─┐
        ├─> agent_loop ──> execute_tool ──> Executor ──> container
tb run ─┘   (agent/loop.py)  (agent/tools.py)  (agent/sandbox.py)
```

`agent_loop` calls the model, executes whatever tools it asks for, feeds the
results back, and repeats until `task_complete`, a turn with no tool calls, or
`max_iterations`. Four tools: `read_file`, `write_file`, `run_shell`,
`task_complete`.

The only thing that differs between the CLI and the benchmark is which `Executor`
gets passed in, so both run identical tool code.

### Notes from building it

- **`execute_tool` never raises.** Llama and Qwen emit malformed tool-call JSON,
  invented tool names and wrong-typed arguments often enough that treating those
  as exceptions would kill a run several times per benchmark. Every failure comes
  back as an error string the model can read and correct.
- **Tool calls written as prose are recovered.** On the very first real run,
  llama-3.3-70b emitted `<function/run_shell {...}</function>` as message *text*
  rather than through the tool-calling API. The loop saw no tool calls and
  stopped on step 1 with the task untouched. `extract_tool_calls_from_text`
  parses the known text formats, and the recovered call is rewritten into the
  transcript in correct structural form. Counted separately as
  `recovered_tool_calls` — it measures the model, not the scaffold.
- **File tools go through the executor, not the host filesystem.** Otherwise the
  benchmark agent would read the host while its shell acts in the container.
- **`task_complete` batched with the work gets deferred.** Models often emit
  write + run + `task_complete` in a single turn, declaring the output verified
  before a single tool result existed. One run wrote bash into a `.py` file, got
  a `SyntaxError`, and claimed success in the same breath — it would have scored
  a false pass. The loop now feeds the results back and requires
  `task_complete` on its own turn.
- **Schemas stay permissive where the dispatcher coerces.** Groq validates tool
  arguments server-side and 400s the whole generation on a mismatch; a model
  sending `"timeout": "10"` killed a run. The rejected text comes back in
  `failed_generation`, so the call is salvaged from it rather than lost.
- **The shell wrapper persists the working directory** between calls. Each
  `docker exec` is a fresh process, so `cd /app` in one command would be silently
  lost by the next.
- **Written file content is base64'd over argv**, so nothing the model generates
  can be reinterpreted as shell syntax. Costs a ~1MB write ceiling (`ARG_MAX`).

## Benchmark

Terminal-Bench needs **Python ≥ 3.12**; the agent itself runs on 3.11+. If your
default interpreter is 3.11, install the harness separately:

```bash
py -3.13 -m pip install terminal-bench
```

Then, with Docker running:

```bash
tb run --dataset terminal-bench-core \
       --agent-import-path adapters.terminal_bench:CliAgent \
       --model llama-3.3-70b-versatile \
       --task-id hello-world
```

Use a fixed ~15–20 task subset while iterating — not the full suite, and not
repeatedly. Groq's free tier is ~1,000 requests/day and each task burns one
request per loop step.

Per-task `metrics.json` and `transcript.json` are written into the harness's
logging directory; they are the input to the failure-mode table below.

### Results

Not yet run — no scores to report.

| | Resolution rate | Avg steps | Avg tokens |
| --- | --- | --- | --- |
| This agent | — | — | — |
| Terminus (reference) | — | — | — |

Failure mode breakdown: _pending first run._

## Status

Built and working end to end: tool dispatch, agent loop, Docker sandbox, CLI,
Terminal-Bench adapter.

First real run, `"write a python script that prints the first 20 primes and run it"`
on `llama-3.3-70b-versatile` — completed in 4 steps / 4658 tokens:

| Step | What happened |
| --- | --- |
| 1 | Tool call arrived as text; recovered and run. The command itself was malformed (literal `\n` inside `python -c "..."`) → `SyntaxError` |
| 2 | Model read the error, switched to `write_file`, and used a proper structured tool call |
| 3 | Ran the script; correct output |
| 4 | `task_complete` |

Not yet done:
- A benchmark run.
- Stretch goal: `spawn_subagent` — a fresh loop with isolated message history
  that returns only its final summary to the parent. `agent_loop` takes an
  `extra_tool_handlers` hook for exactly this, and the hook is tested; the tool
  itself is deliberately left until there is a baseline score to compare against.
