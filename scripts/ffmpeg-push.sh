#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <RTSP_URL> <RTMP_URL>"
  echo "Example:"
  echo "  $0 \"rtsp://pyuser:PASS@192.168.1.70:554/...\" \"rtmp://127.0.0.1:1935/live/camera1\""
  exit 1
fi

RTSP_URL="$1"
RTMP_URL="$2"

# Preferred path is stream copy to avoid CPU-heavy re-encoding.
exec ffmpeg \
  -rtsp_transport tcp \
  -fflags nobuffer \
  -flags low_delay \
  -i "$RTSP_URL" \
  -c copy \
  -f flv \
  "$RTMP_URL"
