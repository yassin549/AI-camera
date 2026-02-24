#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docker compose up -d janus
docker compose ps janus

echo "Janus HTTP API:  http://localhost:8088/janus"
echo "Janus WS API:    ws://localhost:8188"
