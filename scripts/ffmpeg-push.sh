#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RTSP_URL> [JANUS_RTP_URL]"
  echo "Example:"
  echo "  $0 \"rtsp://pyuser:PASS@192.168.1.70:554/...\""
  echo "  $0 \"rtsp://pyuser:PASS@192.168.1.70:554/...\" \"rtp://127.0.0.1:5004?pkt_size=1200\""
  exit 1
fi

RTSP_URL="$1"
JANUS_RTP_URL="${2:-rtp://127.0.0.1:5004?pkt_size=1200}"
REENCODE="${AICAM_FFMPEG_REENCODE:-0}"

# Push camera RTSP to Janus RTP mountpoint (id=1 by default in streaming.jcfg).
# Keep stream copy to avoid CPU-heavy re-encoding.
if [[ "$REENCODE" == "1" || "$REENCODE" == "true" || "$REENCODE" == "yes" ]]; then
  exec ffmpeg \
    -rtsp_transport tcp \
    -fflags nobuffer \
    -flags low_delay \
    -i "$RTSP_URL" \
    -an \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -g 30 \
    -keyint_min 30 \
    -bf 0 \
    -pix_fmt yuv420p \
    -payload_type 96 \
    -f rtp \
    "$JANUS_RTP_URL"
else
  exec ffmpeg \
    -rtsp_transport tcp \
    -fflags nobuffer \
    -flags low_delay \
    -i "$RTSP_URL" \
    -an \
    -c:v copy \
    -payload_type 96 \
    -f rtp \
    "$JANUS_RTP_URL"
fi
