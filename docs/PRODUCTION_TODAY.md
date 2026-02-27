# Production Runbook (Vercel Frontend + Local Backend)

## Target topology
- Frontend on Vercel (React/Vite static app).
- Backend + DB remain local (`main.py --start-api`).
- Camera media path bypasses backend relay:
  - Preferred: RTSP camera -> Janus/WebRTC gateway -> browser.
  - Metadata path: backend `/api/realtime/ws` + `/api/realtime/latest`.

## 1) Backend runtime consistency (local machine)
Use `.env.backend.local.example` as reference:

```bash
API_KEY=replace-with-shared-secret
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:5173,http://127.0.0.1:5173
AICAM_CAPTURE_BUFFER=1
AICAM_WS_MAX_CLIENTS=64
AICAM_WS_HEARTBEAT_SEC=10
AICAM_ENABLE_FRAME_STREAMING=1
```

Start backend with the project `.venv` (avoids missing websocket libs):

```bash
./run_prod.sh
# Windows:
# run_prod.bat
```

## 2) Configure camera direct media path (RTSP)
- Raw `rtsp://` is not browser-safe.
- Use Janus (or another RTSP->WebRTC/HLS gateway) as the media endpoint.
- Keep `id = 1` in `janus/etc/janus/streaming.jcfg` (RTP mountpoint).
- Start Janus:

```bash
docker compose up -d janus
```

Or run Janus + Docker ingest relay together (no local ffmpeg needed):

```bash
docker compose --profile ingest up -d janus ffmpeg-relay
```

Start FFmpeg push into Janus (stream copy):

```bash
scripts/ffmpeg-push.sh "rtsp://USER:PASS@CAMERA_IP:554/your-path"
# Windows:
# scripts\ffmpeg-push.bat "rtsp://USER:PASS@CAMERA_IP:554/your-path"
```

Verify Janus mountpoint/auth quickly:

```bash
python tools/check_janus_mountpoint.py
```

If this prints `janus_mountpoint_error`, inspect `docker compose logs janus` and verify FFmpeg is pushing to Janus RTP port.

## 3) Expose backend publicly with a named tunnel (persistent)
Do this once:

```bash
cloudflared tunnel login
cloudflared tunnel create aicam
cloudflared tunnel route dns aicam api.your-domain.example.com
cloudflared tunnel route dns aicam janus-api.your-domain.example.com
```

Then create `cloudflared/config.yml` from `cloudflared/config.yml.example`, and run:

```bash
./scripts/run-tunnel.sh
# Windows:
# scripts\run-tunnel.bat
```

## 4) Vercel env vars (frontend)
Use `.env.vercel.example` values in Vercel project settings:

```bash
VITE_API_BASE=https://api.your-domain.example.com
VITE_API_KEY=replace-with-shared-secret
VITE_WS_METADATA_URL=wss://api.your-domain.example.com/api/realtime/ws
VITE_METADATA_LATEST_URL=https://api.your-domain.example.com/api/realtime/latest
VITE_METADATA_POLL_MS=250
VITE_JANUS_HTTP_URL=https://janus-api.your-domain.example.com/janus
VITE_JANUS_MOUNTPOINT=1
VITE_DISABLE_BACKEND_VIDEO=true
VITE_DISABLE_WEBRTC=false
```

Optional direct stream URL (non-backend):

```bash
VITE_DIRECT_STREAM_URL=
VITE_DIRECT_STREAM_KIND=auto
```

## 5) Janus internet reachability requirements
- Cloudflare HTTP tunnel exposes Janus signaling only, not RTP/UDP media.
- For non-local viewers, Janus must have internet-reachable ICE candidates.
- Open/forward UDP `10000-10200` from public internet to the Janus host.
- Configure STUN/TURN in Janus when behind NAT, otherwise remote WebRTC can fail even if `/janus` is reachable.

## 6) Deploy frontend to Vercel

```bash
npm run build
vercel --prod
```

## 7) Preflight checks
Run this from local machine after backend + tunnel + Janus are up:

```bash
VITE_API_BASE=https://api.your-domain.example.com \
VITE_API_KEY=replace-with-shared-secret \
python tools/preflight_split_deploy.py

JANUS_HTTP_URL=http://localhost:8088/janus \
JANUS_MOUNTPOINT=1 \
python tools/check_janus_mountpoint.py
```

Expected:
- `/api/health` returns `capture_running=true`.
- `/api/realtime/latest` returns JSON payload.
- WS metadata connects successfully.
- Janus probe returns `janus_mountpoint_ok`.

## 8) Live validation
- Open Vercel URL.
- Live badge should show:
  - `Video: DIRECT` or `Video: JANUS`
  - `Metadata: OPEN`
- Overlay boxes should update continuously.
