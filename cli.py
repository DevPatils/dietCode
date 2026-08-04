"""Run from a checkout without installing: `python cli.py "task"`.

The real entrypoint is agent/cli.py, so that the installed `dietcode` command
and this path execute exactly the same code.
"""

from agent.cli import entrypoint, main  # noqa: F401  (main re-exported for tests)

if __name__ == "__main__":
    entrypoint()
