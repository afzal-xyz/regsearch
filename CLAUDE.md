# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**RegSearch** — a Python project managed with [uv](https://docs.astral.sh/uv/).

Python: `>=3.12` (see `.python-version` / `pyproject.toml`).

## Environment & commands

This project uses **uv** for dependency and environment management. Prefer `uv`
over calling `pip`/`python` directly so the lockfile and virtualenv stay in sync.

| Task | Command |
| --- | --- |
| Sync/create the venv from the lockfile | `uv sync` |
| Add a dependency | `uv add <package>` |
| Add a dev-only dependency | `uv add --dev <package>` |
| Remove a dependency | `uv remove <package>` |
| Run the app | `uv run main.py` |
| Run an arbitrary command in the venv | `uv run <command>` |

## Layout

- `main.py` — entry point.
- `pyproject.toml` — project metadata and dependencies (edited via `uv`).
- `uv.lock` — resolved lockfile; commit it, don't edit by hand.
- `.python-version` — pinned Python version for uv.

## Conventions

- Match the style and structure of surrounding code.
- Keep `pyproject.toml` dependencies changes going through `uv add`/`uv remove`.
- Don't commit secrets; use a `.env` file (already git-ignored).
