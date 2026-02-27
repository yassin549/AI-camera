#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/cloudflared/config.yml"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Missing cloudflared config file: $CONFIG_FILE"
  echo "Copy cloudflared/config.yml.example to cloudflared/config.yml and fill tunnel values first."
  exit 1
fi

cloudflared tunnel --config "$CONFIG_FILE" run

