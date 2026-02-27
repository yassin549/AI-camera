# AIcam Production Setup (Step-by-Step)

This guide is for:
- Backend running locally on your machine (FastAPI + local DB).
- Frontend deployed on Vercel.
- Camera stream delivered to browser through Janus/direct gateway (not through backend relay).

## 1) Do you create the backend API key?
Yes. You create it.

Use one strong random secret and set the same value in:
- Backend env: `API_KEY`
- Vercel env: `VITE_API_KEY`

PowerShell command to generate a strong key:

```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
[Convert]::ToBase64String($bytes)
```

Copy the output and keep it safe.

## 2) Prerequisites
- Python 3.10+ installed
- Node.js 18+ installed
- Docker Desktop installed (for Janus)
- Vercel account + Vercel CLI (`npm i -g vercel`) or Git integration
- One tunnel tool:
  - Cloudflare Tunnel (`cloudflared`)
  - or Tailscale Funnel (`tailscale`)

## 3) Backend local env file
Create a local backend env file from template:

```powershell
Copy-Item .env.backend.local.example .env.backend.local
```

Edit `.env.backend.local` and set at minimum:

```env
API_KEY=PASTE_THE_KEY_YOU_GENERATED
CORS_ORIGINS=https://your-project.vercel.app,http://localhost:5173,http://127.0.0.1:5173
AICAM_CAPTURE_BUFFER=1
AICAM_WS_MAX_CLIENTS=64
AICAM_WS_HEARTBEAT_SEC=10
AICAM_ENABLE_FRAME_STREAMING=1
```

Important:
- Replace `https://your-project.vercel.app` with your real Vercel domain.
- Keep localhost origins while developing locally.

## 4) Install backend deps and run backend
From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Load env vars for current terminal session:

```powershell
Get-Content .env.backend.local | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $name, $value = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
}
```

Start backend:

```powershell
python main.py --config config.yaml --start-api --no-display
```

Keep this terminal running.

## 5) Configure and start Janus (camera media path)
Edit:
- `janus/etc/janus/streaming.jcfg`

Keep mountpoint `id = 1` in the config (RTP mountpoint).

Start Janus:

```powershell
docker compose up -d janus
docker compose ps janus
```

If `ffmpeg` is not installed locally, run Docker-managed ingest relay:

```powershell
docker compose --profile ingest up -d janus ffmpeg-relay
```

Start camera ingest push into Janus:

```powershell
scripts\ffmpeg-push.bat "rtsp://USER:PASS@CAMERA_IP:554/your-path"
```

Why:
- Browsers generally do not play raw `rtsp://` directly.
- Janus converts camera transport to browser-friendly WebRTC.

## 6) Expose backend publicly (for Vercel frontend)
Use a named Cloudflare Tunnel (persistent):

```powershell
cloudflared tunnel login
cloudflared tunnel create aicam
cloudflared tunnel route dns aicam api.your-domain.example.com
cloudflared tunnel route dns aicam janus-api.your-domain.example.com
# copy cloudflared\config.yml.example to cloudflared\config.yml
scripts\run-tunnel.bat
```

Use `https://api.your-domain.example.com` as `VITE_API_BASE`.

## 7) Frontend env values for Vercel
Use `.env.vercel.example` as reference.

In Vercel Project Settings -> Environment Variables, add:

Required:

```env
VITE_API_BASE=https://api.your-domain.example.com
VITE_API_KEY=PASTE_THE_SAME_KEY_AS_BACKEND
VITE_WS_METADATA_URL=wss://api.your-domain.example.com/api/realtime/ws
VITE_METADATA_LATEST_URL=https://api.your-domain.example.com/api/realtime/latest
VITE_METADATA_POLL_MS=250
VITE_JANUS_HTTP_URL=https://janus-api.your-domain.example.com/janus
VITE_JANUS_MOUNTPOINT=1
VITE_DISABLE_BACKEND_VIDEO=true
VITE_DISABLE_WEBRTC=false
```

Optional:

```env
VITE_DIRECT_STREAM_URL=
VITE_DIRECT_STREAM_KIND=auto
VITE_DISABLE_WEBRTC=false
```

Notes:
- Keep `VITE_DISABLE_BACKEND_VIDEO=true` to force video away from backend relay.
- `VITE_API_KEY` must match backend `API_KEY`.
- Raw `rtsp://` cannot be played by browsers; use Janus/HLS/MJPEG gateway URLs.
- Cloudflare HTTP tunnel covers Janus signaling only; remote WebRTC media still requires STUN/TURN + UDP reachability.

## 8) Deploy frontend to Vercel
If using CLI:

```powershell
npm install
npm run build
vercel --prod
```

If using Git integration:
- Push branch
- Ensure env vars are set in Vercel
- Trigger deploy from Vercel dashboard

## 9) Run preflight checks
Use the included checker:

```powershell
$env:VITE_API_BASE="https://api.your-domain.example.com"
$env:VITE_API_KEY="PASTE_THE_SAME_KEY_AS_BACKEND"
python tools/preflight_split_deploy.py

$env:JANUS_HTTP_URL="http://localhost:8088/janus"
$env:JANUS_MOUNTPOINT="1"
python tools/check_janus_mountpoint.py
```

Expected output:
- health payload returned
- `/api/realtime/latest` returned
- metadata WebSocket connected
- `preflight: success`
- `janus_mountpoint_ok`

## 10) Final validation in the app
Open your Vercel app URL and verify:
- Bottom badge shows:
  - `Video: DIRECT` or `Video: JANUS`
  - `Metadata: OPEN`
- Overlay boxes appear and update smoothly.
- Library page loads identities.

## 11) Common issues
`401 Invalid API key`:
- `API_KEY` and `VITE_API_KEY` do not match exactly.

No metadata:
- Wrong `VITE_WS_METADATA_URL`/`VITE_METADATA_LATEST_URL`
- Tunnel down
- Backend not running

No video:
- Janus mountpoint misconfigured
- FFmpeg push not running / wrong RTSP URL in push command
- Browser cannot use raw RTSP (must use Janus/gateway URL)

CORS errors:
- Missing Vercel domain in `CORS_ORIGINS`.
