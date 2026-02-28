# AIcam: Live Person Tracking + Identity Overlay

AIcam is a local-first computer vision stack for live camera streams:

- Python backend captures frames, detects/tracks people, and links tracks to identities.
- FastAPI serves identities, realtime metadata, WebRTC/MJPEG fallbacks, and a Janus proxy.
- React frontend renders video and draws overlays client-side from metadata.

The key design is to keep **video transport separate from CV metadata** for lower latency and lower CPU.

## How It Works

### Runtime pipeline (backend)

`main.py` orchestrates four concurrent parts:

1. **Capture worker** (`capture.py`)
   - Reads RTSP/video frames via OpenCV.
   - Keeps only the newest frame in a single-slot buffer (`LatestFrameStore`) to avoid backlog.

2. **Detection + tracking loop** (`detector_yolo.py`, `tracker_adapter.py`)
   - Runs YOLOv8 ONNX person detection.
   - Updates track state (ByteTrack when available, fallback IoU tracker otherwise).
   - Uses adaptive detection interval to maintain target FPS on CPU.

3. **Recognition worker** (`recognition_worker.py`)
   - Face-first matching (MediaPipe ROI + face embedding + cosine match in SQLite-backed index).
   - Body fallback matching after repeated face misses (optional).
   - Creates/updates identities and sample images in `faces/`.

4. **API server thread** (`api_server.py` + `api/*`)
   - Publishes realtime metadata (`/api/realtime/ws`, `/api/realtime/latest`).
   - Serves identities (`/api/identities`) and media (`/media/*`).
   - Offers backend WebRTC fallback (`/webrtc/offer`) and MJPEG (`/api/media/mjpeg`).
   - Proxies Janus HTTP signaling (`/janus/*`) to upstream Janus.

### Frontend media + metadata model

`src/components/VideoCanvas.tsx` uses independent channels:

- **Video channel** (best-effort fallback order):
  1. Direct stream URL (if configured)
  2. Janus WebRTC
  3. Backend WebRTC (`/webrtc/offer`)
  4. Backend MJPEG (`/api/media/mjpeg`)

- **Metadata channel**:
  - Primary: WebSocket `/api/realtime/ws`
  - Bootstrap/fallback: polling `/api/realtime/latest`

The overlay is always drawn on a `<canvas>` over the video element, so boxes/labels stay responsive even if video source changes.

## Repo Map

Core files you will likely touch first:

- `main.py`: backend orchestrator and CLI entrypoint.
- `config.yaml`: runtime pipeline config (camera URL, thresholds, FPS, API host/port).
- `api_server.py`: FastAPI app wiring and shared runtime state.
- `api/realtime.py`: metadata WS + backend WebRTC offer route.
- `api/media.py`: MJPEG stream + static media mount.
- `api/identities.py`: identity CRUD + reindexing.
- `src/pages/LiveView.tsx`: live UI.
- `src/components/VideoCanvas.tsx`: transport fallback + overlay drawing.
- `docker-compose.yml`: Janus gateway + optional FFmpeg relay.

`api.py` exists but is a legacy minimal API path. The active runtime path is `main.py` + `api_server.py`.

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm
- Docker Desktop (optional, for Janus)
- `ffmpeg` (optional if you use local RTSP->RTP push script)

## Run Locally

### 1) Configure camera source

Edit `config.yaml` and set `rtsp_url` to your camera or stream source.

Minimum keys to verify:

- `rtsp_url`
- `imgsz`
- `http_host`
- `http_port`
- `db_path`

### 2) Install backend dependencies

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Optional backend env file (API key, CORS, tuning)

Copy and edit:

- `.env.backend.local.example` -> `.env.backend.local`

If you use `run_prod.bat` / `run_prod.sh`, this env file is loaded automatically.

### 4) Start backend

Recommended (loads `.env.backend.local`, no display window):

Windows:

```powershell
.\run_prod.bat
```

Linux/macOS:

```bash
./run_prod.sh
```

Alternative direct run:

```bash
python main.py --config config.yaml --start-api --no-display
```

### 5) Start frontend

```bash
npm install
npm run dev
```

Open: `http://localhost:5173`

### 6) Quick health checks

```bash
curl http://127.0.0.1:8080/api/health
curl http://127.0.0.1:8080/api/realtime/latest
curl http://127.0.0.1:8080/api/identities
```

## Janus Setup (Recommended for RTSP Cameras)

Browsers do not play raw `rtsp://` directly. Use Janus (or another gateway) for browser-safe video.

### Start Janus

```bash
docker compose up -d janus
docker compose ps janus
```

### Configure mountpoint

Edit `janus/etc/janus/streaming.jcfg`.
Default mountpoint in this repo is `id = 1`.

### Push RTSP into Janus

Linux/macOS:

```bash
scripts/ffmpeg-push.sh "rtsp://USER:PASS@CAMERA_IP:554/your-path"
```

Windows:

```powershell
scripts\ffmpeg-push.bat "rtsp://USER:PASS@CAMERA_IP:554/your-path"
```

Optional Docker-managed ingest (no local ffmpeg install):

```bash
docker compose --profile ingest up -d janus ffmpeg-relay
```

Set `JANUS_RTSP_URL` in `.env` first when using this profile.

### Validate Janus signaling/mountpoint

```bash
python tools/check_janus_mountpoint.py
```

## Configuration

### `config.yaml` highlights

- `rtsp_url`: input stream URL.
- `imgsz`: processing resolution used by capture + detector pipeline.
- `target_fps`: used for adaptive scheduling.
- `detection_interval`, `detection_interval_max`, `adaptive_detection_interval`.
- `face_interval_frames`, `body_interval_frames`.
- `face_threshold`, `body_threshold`.
- `max_tracks`, `max_track_age_frames`.
- `http_host`, `http_port`.
- `yolo_onnx_path`: person detector ONNX path.

### Backend env vars

- `API_KEY`: enables auth on identities/realtime/media/janus routes.
- `CORS_ORIGINS`, `CORS_ORIGIN_REGEX`: browser access control.
- `AICAM_ENABLE_FRAME_STREAMING`: enables backend WebRTC/MJPEG frame streaming.
- `AICAM_MJPEG_FPS`, `AICAM_MJPEG_QUALITY`, `AICAM_MJPEG_MAX_CLIENTS`.
- `AICAM_CAPTURE_BUFFER`: capture queue size (low value reduces latency).
- `AICAM_WS_HEARTBEAT_SEC`, `AICAM_WS_MAX_CLIENTS`.
- `AICAM_JANUS_UPSTREAM`: upstream Janus URL for `/janus` proxy.
- `AICAM_STATIC_PATH`: media root for `/media`.
- `AICAM_DELETE_SAMPLE_FILES`: delete files on identity delete.

### Frontend env vars

- `VITE_API_BASE`
- `VITE_API_KEY`
- `VITE_WS_METADATA_URL`
- `VITE_METADATA_LATEST_URL`
- `VITE_METADATA_POLL_MS`
- `VITE_JANUS_HTTP_URL`
- `VITE_JANUS_MOUNTPOINT`
- `VITE_DIRECT_STREAM_URL`
- `VITE_DIRECT_STREAM_KIND` (`auto|video|image`)
- `VITE_DISABLE_JANUS`
- `VITE_DISABLE_BACKEND_VIDEO`
- `VITE_DISABLE_WEBRTC`

## API Endpoints

- `GET /api/health` (also `/health`, `/healthz`)
- `GET /metrics`
- `GET /api/identities`
- `GET /api/identities/{id}`
- `POST /api/identities/{id}/rename`
- `DELETE /api/identities/{id}`
- `POST /api/identities/reindex`
- `GET /api/realtime/latest`
- `WS /api/realtime/ws`
- `POST /webrtc/offer`
- `GET /api/media/mjpeg`
- `GET|POST /janus/*` (reverse proxy)

## Vercel Frontend + ngrok Backend (Required Deployment Mode)

This project supports a single public endpoint topology:

- Frontend is deployed on Vercel.
- Backend stays local (`main.py --start-api`).
- ngrok exposes local backend `:8080` to the public internet.

### 1) Start backend locally

Windows:

```powershell
.\run_prod.bat
```

Linux/macOS:

```bash
./run_prod.sh
```

### 2) Start ngrok tunnel to backend port 8080

One-time auth:

```bash
ngrok config add-authtoken <YOUR_NGROK_TOKEN>
```

Start tunnel (free random domain):

```bash
ngrok http 8080
```

Or reserved domain:

```bash
ngrok http --domain=<your-name>.ngrok-free.app 8080
```

Use the resulting `https://...ngrok-free.app` as your public backend URL.

### 3) Backend env (`.env.backend.local`)

Use `.env.backend.local.example` and set:

- `API_KEY=<shared-secret>`
- `CORS_ORIGINS=https://<your-vercel-app>.vercel.app,http://localhost:5173,http://127.0.0.1:5173`
- `AICAM_JANUS_UPSTREAM=http://127.0.0.1:8088/janus` (if you use Janus proxy path)

### 4) Vercel frontend env vars

In Vercel Project Settings -> Environment Variables (see `.env.vercel.example`):

```env
VITE_API_BASE=https://<your-ngrok-domain>.ngrok-free.app
VITE_API_KEY=<same-shared-secret-as-API_KEY>
VITE_JANUS_HTTP_URL=/janus
VITE_DISABLE_JANUS=true
VITE_DISABLE_BACKEND_VIDEO=false
VITE_DISABLE_WEBRTC=false
```

Notes:

- Leave `VITE_WS_METADATA_URL` and `VITE_METADATA_LATEST_URL` empty to auto-derive from `VITE_API_BASE`.
- `API_KEY` and `VITE_API_KEY` must match exactly.
- After changing Vercel env vars, redeploy frontend.

### 5) Validate end-to-end

Run preflight against ngrok URL:

```bash
VITE_API_BASE=https://<your-ngrok-domain>.ngrok-free.app VITE_API_KEY=<shared-secret> python tools/preflight_split_deploy.py
```

Then open your Vercel URL and verify:

- Metadata badge becomes `OPEN`.
- Video badge shows `WEBRTC` or `MJPEG`.
- Overlay boxes update on live view.

## Troubleshooting

- No video in browser:
  - Raw RTSP is not browser-safe; use Janus/gateway URL.
  - Check Janus mountpoint and FFmpeg push.
  - Inspect frontend badge for transport mode and error text.

- Metadata missing:
  - Verify `/api/realtime/ws` and `/api/realtime/latest`.
  - Confirm API key matches (`x-api-key` / `api_key`).
  - Verify `VITE_API_BASE` points to the active ngrok URL (it changes when not reserved).

- High backend CPU:
  - Prefer Janus/direct stream path for video.
  - Keep `AICAM_CAPTURE_BUFFER=1`.
  - Tune `detection_interval*` and `imgsz`.

- ngrok free plan + MJPEG issues:
  - ngrok browser warning pages can break direct image stream usage.
  - Prefer WebRTC path when possible.

- Face detection unavailable warning:
  - The app still runs, but recognition quality drops.
  - Check `mediapipe` installation and backend logs.

## Useful Commands

- Frontend tests: `npm test`
- Python tests: `pytest`
- Verify API quickly: `python tools/verify_api.py`
- Janus probe: `python tools/check_janus_mountpoint.py`
- Split deploy preflight: `python tools/preflight_split_deploy.py`
