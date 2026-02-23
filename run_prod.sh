#!/usr/bin/env bash
set -euo pipefail

python main.py --headless --use-waitress --verbose "$@"
