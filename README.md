# AIcam: Janus-first Live Video + Metadata Overlay

AIcam now uses a media-gateway architecture:
- Browser video path: camera RTSP/H264 -> Janus WebRTC -> `<video>` (hardware decode).
- Python path: detection/tracking/identity only -> metadata WebSocket (`/api/realtime/ws`).
- Overlay path: metadata WS -> canvas draw over video.

Live video transport order is:
1. Janus WebRTC gateway
2. API WebRTC (`/webrtc/offer`) fallback
3. MJPEG (`/api/media/mjpeg`) fallback

## Why this architecture
- Python JPEG/MJPEG encode loops are CPU-heavy and add latency.
- WS-JPEG full-frame transport is bandwidth-heavy and jittery.
- Relaying already encoded H264 through WebRTC keeps latency low and CPU usage much lower on i5-class CPUs.

## Files added for gateway setup
- `docker-compose.yml`
- `janus/etc/janus/streaming.jcfg`
- `janus/test-player.html`
- `scripts/start-janus.sh`
- `scripts/ffmpeg-push.sh`

## Run locally
1. Start Janus:
```bash
docker compose up -d janus
```
If you want Docker-managed camera ingest (no local ffmpeg install), set `JANUS_RTSP_URL` in `.env` and run:
```bash
docker compose --profile ingest up -d janus ffmpeg-relay
```

2. Configure Janus mountpoint:
- Edit `janus/etc/janus/streaming.jcfg`.
- Keep mountpoint `id = 1` (default in this repo).
- Start FFmpeg RTP push into Janus from your RTSP camera:
```bash
scripts/ffmpeg-push.sh "rtsp://USER:PASS@CAMERA_IP:554/your-path"
```
Windows:
```powershell
scripts\ffmpeg-push.bat "rtsp://USER:PASS@CAMERA_IP:554/your-path"
```

3. Start Python backend:
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --config config.yaml --start-api
```

4. Start frontend:
```bash
npm i
npm run dev
```

5. Open:
- `http://localhost:5173`
- Live page should show `Video: JANUS` in the status badge.

## Frontend environment (optional)
- `VITE_API_BASE` default: `http://localhost:8080`
- `VITE_API_KEY` optional shared key sent as `x-api-key` (HTTP) and `api_key` (WS/media query)
- `VITE_WS_METADATA_URL` default: `ws://localhost:8080/api/realtime/ws`
- `VITE_METADATA_LATEST_URL` default: `${VITE_API_BASE}/api/realtime/latest` (used for fast bootstrap/fallback when WS reconnects)
- `VITE_METADATA_POLL_MS` default: `250` (poll cadence used only when WS is not open)
- `VITE_WEBRTC_OFFER` default: `http://localhost:8080/webrtc/offer`
- `VITE_JANUS_HTTP_URL` default: `http://localhost:8088/janus`
- `VITE_JANUS_MOUNTPOINT` default: `1`
- `VITE_DIRECT_STREAM_URL` optional direct media URL (for example camera MJPEG, HLS, or WebRTC gateway URL)
- `VITE_DIRECT_STREAM_KIND` one of `auto|video|image` (default `auto`)
- `VITE_DISABLE_BACKEND_VIDEO=true` blocks API WebRTC/MJPEG fallbacks so media does not relay through backend
- `VITE_DISABLE_WEBRTC=true` skips Janus/API WebRTC and uses MJPEG fallback only.

Important:
- Standard web browsers do not decode raw `rtsp://` URLs directly.
- For RTSP cameras, use Janus (or another RTSP->WebRTC/HLS gateway) and point frontend to that gateway URL.

## Backend environment (optional)
- `API_KEY` enables shared-key protection for identities/realtime/media/WebRTC endpoints.
- `CORS_ORIGINS` comma-separated allowlist used by FastAPI CORS middleware (for example Vercel + localhost).
- `AICAM_ENABLE_FRAME_STREAMING=1` enables MJPEG fallback endpoints (default enabled).
- `AICAM_MJPEG_FPS` / `AICAM_MJPEG_QUALITY` tune fallback CPU/bandwidth tradeoffs.
- `AICAM_CAPTURE_BUFFER=1` keeps capture queue short for lower end-to-end latency.

## Split deployment (Vercel frontend + local backend)
Target topology:
- Frontend (Vite/React) is deployed on Vercel.
- Backend (FastAPI + CV pipeline) stays local.
- Browser calls backend through a public tunnel endpoint.

Quick runbook:
- `docs/PRODUCTION_TODAY.md`
- `docs/SETUP_FRONTEND_BACKEND_STEP_BY_STEP.md`
- `.env.vercel.example`
- `.env.backend.local.example`

### 1) Start backend locally
```bash
./run_prod.sh
# Windows:
# run_prod.bat
```

### 2) Expose backend with a tunnel
Recommended: named Cloudflare Tunnel (persistent hostname):

```bash
cloudflared tunnel login
cloudflared tunnel create aicam
cloudflared tunnel route dns aicam api.your-domain.example.com
cloudflared tunnel route dns aicam janus-api.your-domain.example.com
# then copy cloudflared/config.yml.example -> cloudflared/config.yml
# and run scripts/run-tunnel.sh (or scripts\run-tunnel.bat on Windows)
```

Use `https://api.your-domain.example.com` as your backend base.

### 3) Backend env vars (local machine)
Set these before starting Python:
```bash
API_KEY=your-shared-secret
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:5173,http://127.0.0.1:5173
```

Notes:
- `CORS_ORIGINS` is read by `config.py` and passed to FastAPI CORS middleware.
- Keep your Vercel origin in `CORS_ORIGINS`, and keep localhost origins if you still use local frontend dev.

### 4) Vercel env vars (frontend project)
Set in Vercel Project Settings -> Environment Variables:
```bash
VITE_API_BASE=https://api.your-domain.example.com
VITE_API_KEY=your-shared-secret
```

Optional overrides:
```bash
VITE_WS_METADATA_URL=wss://api.your-domain.example.com/api/realtime/ws
VITE_METADATA_LATEST_URL=https://api.your-domain.example.com/api/realtime/latest
VITE_METADATA_POLL_MS=250
VITE_WEBRTC_OFFER=https://api.your-domain.example.com/webrtc/offer
VITE_JANUS_HTTP_URL=https://janus-api.your-domain.example.com/janus
VITE_JANUS_MOUNTPOINT=1
VITE_DIRECT_STREAM_URL=
VITE_DIRECT_STREAM_KIND=auto
VITE_DISABLE_BACKEND_VIDEO=true
VITE_DISABLE_WEBRTC=false
```

Notes for media routing:
- Keep `VITE_DISABLE_BACKEND_VIDEO=true` when you want video to stay gateway/direct only.
- If you provide `VITE_DIRECT_STREAM_URL`, frontend tries it first.
- For RTSP cameras, use a gateway URL (Janus/HLS/WebRTC). Raw `rtsp://` is not browser-safe.
- Cloudflare HTTP tunnel exposes Janus signaling only; remote WebRTC media still needs reachable ICE candidates (STUN/TURN + UDP).

If `VITE_API_KEY` is set, the frontend will:
- Send `x-api-key` on HTTP requests.
- Append `api_key` to WS/media URLs where headers are unavailable in browser primitives.

### 5) Deploy frontend to Vercel
This repo includes `vercel.json` with SPA rewrite to `index.html`.

CLI flow:
```bash
npm run build
vercel
```

Or connect the Git repository in Vercel and set the same environment variables there.

## Janus live-view integration sample
Use this pattern in `src/pages/LiveView.jsx` / `src/components/VideoCanvas.tsx`:

```js
const janusServer = "http://localhost:8088/janus";
const mountpointId = 1;

async function janusPost(path, body) {
  const response = await fetch(`${janusServer}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, transaction: Math.random().toString(36).slice(2, 10) })
  });
  if (!response.ok) throw new Error(`Janus HTTP ${response.status}`);
  return await response.json();
}

async function startJanus(videoEl) {
  const create = await janusPost("", { janus: "create" });
  const sessionId = create.data.id;
  const attach = await janusPost(`/${sessionId}`, {
    janus: "attach",
    plugin: "janus.plugin.streaming"
  });
  const handleId = attach.data.id;

  const pc = new RTCPeerConnection();
  pc.ontrack = (ev) => {
    videoEl.srcObject = ev.streams[0];
    videoEl.play();
  };
  pc.onicecandidate = (ev) => {
    janusPost(`/${sessionId}/${handleId}`, {
      janus: "trickle",
      candidate: ev.candidate ?? { completed: true }
    }).catch(() => {});
  };

  await janusPost(`/${sessionId}/${handleId}`, {
    janus: "message",
    body: { request: "watch", id: mountpointId }
  });

  // Poll Janus events, apply remote offer, create answer, then send "start".
}
```

## API endpoints
- `GET /api/health` / `GET /health` -> health + capabilities (`face_detector_available`, `processing_resolution`, etc.)
- `GET /metrics` -> runtime counters/gauges (`face_attempts`, `face_detections`, `embedding_success`, `db_writes`, `active_tracks`, ...)
- `GET /api/identities` -> identity list JSON
- `GET /api/realtime/latest` -> latest metadata payload
- `WS /api/realtime/ws` -> metadata stream

## Pipeline tuning keys (`config.yaml`)
- `detection_interval` (default `1`)
- `adaptive_detection_interval` (default `true`)
- `detection_interval_max` (default `6`)
- `target_fps` (default `30`)
- `max_track_age_frames` (default `24`)
- `face_interval_frames` (default `4`)
- `face_top_ratio` (default `0.68`)
- `face_min_pixels` (default `40`)
- `body_fallback_enabled` (default `true`)
- `body_fallback_after_face_failures` (default `3`)
- `body_fallback_model_path` (default empty -> color-hist fallback)

## Verification checklist
- Janus container is healthy: `docker compose ps janus`
- Janus player test works: open `janus/test-player.html` and see live video.
- Frontend Live page shows `Video: JANUS`.
- Boxes and IDs render in overlay from metadata WS.
- `curl http://127.0.0.1:8080/api/health` returns status JSON.
- `curl http://127.0.0.1:8080/api/identities` returns JSON list.
- Stop Janus: frontend should automatically fall back to API WebRTC or MJPEG.

## Debug checklist
- No Janus video: verify `VITE_JANUS_HTTP_URL` and `VITE_JANUS_MOUNTPOINT`; UI should still fall back automatically.
- No overlay: verify metadata WS at `/api/realtime/ws`.
- High backend CPU: verify Janus is ingesting a direct H264 stream (`-c copy`) and avoid re-encode hops.
- Janus mountpoint/auth issues: run `python tools/check_janus_mountpoint.py` and inspect `docker compose logs janus`.
- Camera ingest issues: run `scripts/ffmpeg-push.sh` (or `scripts\ffmpeg-push.bat`) with `-c copy` to feed Janus RTP.
