#!/usr/bin/env bash
set -euo pipefail

python main.py --config config.yaml --start-api --no-display "$@"
