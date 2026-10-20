#!/usr/bin/env bash
# Quality gate: formatting, lint, types, fast tests.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff format --check .
uv run ruff check .
uv run mypy src/netdesign
uv run pytest -m "not slow" -q
