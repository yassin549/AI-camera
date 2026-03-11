#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8080}"

if ! command -v ngrok >/dev/null 2>&1; then
  echo "[ERROR] ngrok is not installed or not on PATH."
  echo "Install ngrok and run: ngrok config add-authtoken <YOUR_NGROK_TOKEN>"
  exit 1
fi

echo "[INFO] Starting ngrok HTTP tunnel on port ${PORT}"
ngrok http "${PORT}"
