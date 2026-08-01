#!/usr/bin/env bash
#
# Run the Terminal-Bench harness against this agent.
#
# Must run on Linux (or WSL on Windows). terminal-bench 0.2.18 builds *container*
# paths with pathlib, so on Windows "/tmp" becomes "\tmp" and the harness dies in
# TmuxSession.__init__ with `404 Could not find the file \tmp` -- before the agent
# is ever invoked. Its dataset downloader also shells out to Unix `rm -rf .git`.
#
#   wsl bash scripts/benchmark.sh                    # hello-world
#   wsl bash scripts/benchmark.sh fix-git            # one task by id
#   wsl bash scripts/benchmark.sh subset             # the fixed scoring subset
#   wsl bash scripts/benchmark.sh ""                 # all 81 tasks (don't, while iterating)
#
# Requires Docker running and GROQ_API_KEY in .env.
set -u

TASK_ID="${1-hello-world}"
DATASET="${DATASET:-terminal-bench-core==0.1.1}"
MODEL="${MODEL:-llama-3.3-70b-versatile}"
CONCURRENCY="${CONCURRENCY:-2}"

# A fixed subset, per plannings.md: never iterate against all 81 tasks. Chosen to
# spread across categories (sysadmin, security, data, SWE, plain coding) while
# skipping the multi-hour builds -- kernel/qemu compiles, model training and
# dataset downloads say more about your network than about the agent.
SUBSET=(
  hello-world                 # sanity
  fix-git                     # git
  fix-permissions             # sysadmin
  configure-git-webserver     # sysadmin + config
  nginx-request-logging       # config parsing
  csv-to-parquet              # data wrangling
  heterogeneous-dates         # data cleaning
  sqlite-db-truncate          # databases
  openssl-selfsigned-cert     # security / tooling
  crack-7z-hash.easy          # security
  fibonacci-server            # write a service
  simple-web-scraper          # scripting
  organization-json-generator # structured output
  write-compressor            # algorithms
  grid-pattern-transform      # algorithms
  swe-bench-langcodes         # real-world bug fix
)

# The registry's "head" dataset is broken: it points at ./tasks, but the repo
# moved to harbor-framework/terminal-bench and renamed that to original-tasks/.
# Pinning a version avoids it.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$PROJECT_DIR" || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Run output goes to the Linux filesystem, never the Windows-mounted project
# dir. A OneDrive-backed folder over WSL's drvfs is not a reliable filesystem:
# mkdir returns FileNotFoundError then FileExistsError for the same path, and
# os.getcwd() intermittently throws. Both killed earlier runs mid-flight. It is
# also far faster than drvfs. Results are summarised back into the project.
RUNS_ROOT="${RUNS_ROOT:-$HOME/tb-runs}"
mkdir -p "$RUNS_ROOT"

if ! command -v tb >/dev/null 2>&1; then
  echo "terminal-bench not found. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  uv tool install terminal-bench"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker is not reachable -- is Docker Desktop running (with WSL integration)?"
  exit 1
fi

# tb does not read .env. The adapter loads it too, but export here as well in
# case tb's isolated environment has no python-dotenv.
if [ -f .env ] && [ -z "${GROQ_API_KEY:-}" ]; then
  export GROQ_API_KEY="$(grep -E '^GROQ_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d '[:space:]')"
fi
if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "GROQ_API_KEY is not set (put it in .env)"
  exit 1
fi

# Absolute, deliberately. tb finishes by printing output_path.absolute(), and on
# a relative path that calls os.getcwd() -- which throws if the working directory
# has become unreadable (OneDrive-backed folders over WSL's drvfs do this). The
# run itself succeeds and then the harness dies printing its own summary.
run_one() {
  local task="$1" out="$2"
  # One task per invocation, and no --n-concurrent-trials. Passing that flag
  # makes `docker compose build` fail immediately for every task (exit 17)
  # before the agent runs -- 16/16 failures that look exactly like a 0% score.
  tb run \
    --dataset "$DATASET" \
    --agent-import-path adapters.terminal_bench:CliAgent \
    --model "$MODEL" \
    --task-id "$task" \
    --output-path "$out"
}

if [ "$TASK_ID" = "subset" ]; then
  OUT="$RUNS_ROOT/subset"
  rm -rf "$OUT"
  echo "dataset $DATASET | model $MODEL | ${#SUBSET[@]} tasks, one at a time"
  echo "output $OUT"
  for t in "${SUBSET[@]}"; do
    echo ""
    echo "=============== $t ==============="
    run_one "$t" "$OUT/$t" || echo "!! $t: harness error"
  done
  echo ""
  echo "=============== summary ==============="
  python3 "$PROJECT_DIR/scripts/summarize.py" "$OUT"
elif [ -n "$TASK_ID" ]; then
  echo "dataset $DATASET | model $MODEL | task $TASK_ID"
  run_one "$TASK_ID" "$RUNS_ROOT/$TASK_ID"
else
  echo "dataset $DATASET | model $MODEL | ALL tasks"
  tb run --dataset "$DATASET" \
    --agent-import-path adapters.terminal_bench:CliAgent \
    --model "$MODEL" \
    --output-path "$RUNS_ROOT/full"
fi
