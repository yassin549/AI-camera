# Production Runbook (Vercel Frontend + Local Backend)

## Target topology
- Frontend on Vercel (React/Vite static app).
- Backend + DB remain local (`main.py --start-api`).
- Camera media path bypasses backend relay:
  - Preferred: RTSP camera -> Janus/WebRTC gateway -> browser.
  - Metadata path: backend `/api/realtime/ws` + `/api/realtime/latest`.

## 1) Backend env (local machine)
Use `.env.backend.local.example` values:

```bash
API_KEY=replace-with-shared-secret
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:5173,http://127.0.0.1:5173
AICAM_CAPTURE_BUFFER=1
AICAM_WS_MAX_CLIENTS=64
AICAM_WS_HEARTBEAT_SEC=10
AICAM_ENABLE_FRAME_STREAMING=1
```

Start backend:

```bash
python main.py --config config.yaml --start-api --no-display
```

## 2) Configure camera direct media path (RTSP)
- Raw `rtsp://` is not browser-safe.
- Use Janus (or another RTSP->WebRTC/HLS gateway) as the media endpoint.
- Update `janus/etc/janus/streaming.jcfg` with your camera RTSP URL.
- Start Janus:

```bash
docker compose up -d janus
```

## 3) Expose backend publicly
Use Cloudflare Tunnel or Tailscale Funnel:

```bash
cloudflared tunnel --url http://localhost:8080
```

Take the generated HTTPS URL as `VITE_API_BASE`.

## 4) Vercel env vars
Use `.env.vercel.example` values in Vercel project settings:

```bash
VITE_API_BASE=https://your-backend-public-url.example.com
VITE_API_KEY=replace-with-shared-secret
VITE_WS_METADATA_URL=wss://your-backend-public-url.example.com/api/realtime/ws
VITE_METADATA_LATEST_URL=https://your-backend-public-url.example.com/api/realtime/latest
VITE_METADATA_POLL_MS=250
VITE_JANUS_HTTP_URL=https://your-janus-endpoint.example.com/janus
VITE_JANUS_MOUNTPOINT=1
VITE_DISABLE_BACKEND_VIDEO=true
```

Optional direct stream URL (non-backend):

```bash
VITE_DIRECT_STREAM_URL=
VITE_DIRECT_STREAM_KIND=auto
```

## 5) Deploy frontend to Vercel

```bash
npm run build
vercel --prod
```

## 6) Preflight checks
Run this from local machine after tunnel + backend are up:

```bash
VITE_API_BASE=https://your-backend-public-url.example.com \
VITE_API_KEY=replace-with-shared-secret \
python tools/preflight_split_deploy.py
```

Expected:
- `/api/health` returns `capture_running=true`.
- `/api/realtime/latest` returns JSON payload.
- WS metadata connects successfully.

## 7) Live validation
- Open Vercel URL.
- Live badge should show:
  - `Video: DIRECT` or `Video: JANUS`
  - `Metadata: OPEN`
- Overlay boxes should update continuously.

