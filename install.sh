#!/usr/bin/env bash
# netdesign installer: Python environment (.venv) with all dependencies, NonLinLoc, checks.
#
# Usage: ./install.sh [options]
#   --nll-dir DIR   use the NonLinLoc programs in DIR (Vel2Grid, Grid2Time, NLLoc)
#   --global        also make `netdesign` a command available everywhere (uv tool)
#   --test          run the test suite at the end
#   --yes           answer yes to prompts (e.g. installing uv)
#   -h, --help      show this help
#
# Without --nll-dir, NonLinLoc is looked for on PATH. NonLinLoc itself is not installed:
# build it from https://github.com/alomax/NonLinLoc and give its bin directory.
# Requirements: Linux or macOS; curl (only to install uv if missing).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NLL_PROGRAMS=(Vel2Grid Grid2Time NLLoc)

NLL_DIR=""
GLOBAL=0
RUN_TESTS=0
YES=0

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nll-dir) [[ $# -ge 2 ]] || die "--nll-dir needs a directory"; NLL_DIR="$2"; shift 2 ;;
    --global) GLOBAL=1; shift ;;
    --test) RUN_TESTS=1; shift ;;
    --yes|-y) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

confirm() {  # confirm "question" -> 0 if yes
  [[ $YES -eq 1 ]] && return 0
  [[ -t 0 ]] || die "$1 — rerun with --yes to allow this non-interactively"
  read -r -p "$1 [y/N] " answer
  [[ "$answer" =~ ^[Yy] ]]
}

has_nll() {  # has_nll DIR -> 0 if all programs are there and executable
  local d="$1" p
  for p in "${NLL_PROGRAMS[@]}"; do [[ -x "$d/$p" ]] || return 1; done
}

# ------------------------------------------------------------------ 1. uv
say "Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    [[ -x "$candidate" ]] && export PATH="$(dirname "$candidate"):$PATH" && break
  done
fi
if ! command -v uv >/dev/null 2>&1; then
  confirm "uv (Python package manager, https://docs.astral.sh/uv/) is not installed. Install it?" \
    || die "uv is required; install it from https://docs.astral.sh/uv/ and rerun"
  command -v curl >/dev/null 2>&1 || die "curl is needed to install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not found on PATH; open a new shell and rerun"
fi
echo "uv $(uv --version | awk '{print $2}')"

# ------------------------------------------------------------------ 2. venv + dependencies
say "Creating .venv and installing dependencies (uv downloads Python 3.11/3.12 if needed)"
cd "$ROOT"
uv sync --all-extras
uv run netdesign --help >/dev/null || die "netdesign did not install correctly"
echo "environment: $ROOT/.venv"

# ------------------------------------------------------------------ 3. NonLinLoc
say "NonLinLoc"
if [[ -z "$NLL_DIR" ]] && command -v Vel2Grid >/dev/null 2>&1; then
  NLL_DIR="$(dirname "$(command -v Vel2Grid)")"
fi
if [[ -n "$NLL_DIR" ]]; then
  NLL_DIR="$(cd "$NLL_DIR" && pwd)"
  has_nll "$NLL_DIR" || die "$NLL_DIR does not contain executable ${NLL_PROGRAMS[*]}"
  echo "found NonLinLoc programs in $NLL_DIR"
else
  warn "NonLinLoc not found. Rerun with --nll-dir DIR, or set nll.bin_dir in your config."
fi

# ------------------------------------------------------------------ 4. global command
if [[ $GLOBAL -eq 1 ]]; then
  say "Installing the netdesign command globally (uv tool, editable)"
  uv tool install --editable --force "$ROOT"
  uv tool update-shell >/dev/null 2>&1 || true
fi

# ------------------------------------------------------------------ 5. tests
if [[ $RUN_TESTS -eq 1 ]]; then
  say "Running tests"
  if [[ -n "$NLL_DIR" ]]; then
    NETDESIGN_NLL_BIN="$NLL_DIR" uv run pytest -q
  else
    uv run pytest -q  # NonLinLoc tests are skipped
  fi
fi

# ------------------------------------------------------------------ summary
say "Done"
cat <<EOF

To run netdesign:
EOF
if [[ $GLOBAL -eq 1 ]]; then
  echo "  netdesign run config.yaml            (open a new shell first if the command is not found)"
else
  echo "  source $ROOT/.venv/bin/activate"
  echo "  netdesign run config.yaml"
fi
cat <<EOF

Start a project from $ROOT/templates/config.yaml.
EOF
if [[ -n "$NLL_DIR" ]]; then
  cat <<EOF
In your config, point at NonLinLoc with:
  nll:
    bin_dir: $NLL_DIR
EOF
fi
