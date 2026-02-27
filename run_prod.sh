#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "[ERROR] Virtualenv Python not found at $PYTHON_EXE"
  echo "Create it first: python -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

ENV_FILE="$ROOT_DIR/.env.backend.local"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

"$PYTHON_EXE" main.py --config config.yaml --start-api --no-display "$@"
