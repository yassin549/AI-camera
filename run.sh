#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.yaml}"
shift || true

python main.py --config "$CONFIG" --start-api "$@"
