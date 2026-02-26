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

Set your RTSP camera URL in the mountpoint config.

Start Janus:

```powershell
docker compose up -d janus
docker compose ps janus
```

Why:
- Browsers generally do not play raw `rtsp://` directly.
- Janus converts camera transport to browser-friendly WebRTC.

## 6) Expose backend publicly (for Vercel frontend)
In a new terminal, run one of:

Cloudflare Tunnel:

```powershell
cloudflared tunnel --url http://localhost:8080
```

or Tailscale Funnel:

```powershell
tailscale funnel 8080
```

Copy the generated public HTTPS URL, for example:
- `https://abcd1234.trycloudflare.com`

You will use this as `VITE_API_BASE`.

## 7) Frontend env values for Vercel
Use `.env.vercel.example` as reference.

In Vercel Project Settings -> Environment Variables, add:

Required:

```env
VITE_API_BASE=https://YOUR_PUBLIC_BACKEND_URL
VITE_API_KEY=PASTE_THE_SAME_KEY_AS_BACKEND
VITE_WS_METADATA_URL=wss://YOUR_PUBLIC_BACKEND_HOST/api/realtime/ws
VITE_METADATA_LATEST_URL=https://YOUR_PUBLIC_BACKEND_HOST/api/realtime/latest
VITE_METADATA_POLL_MS=250
VITE_JANUS_HTTP_URL=https://YOUR_JANUS_ENDPOINT/janus
VITE_JANUS_MOUNTPOINT=1
VITE_DISABLE_BACKEND_VIDEO=true
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
$env:VITE_API_BASE="https://YOUR_PUBLIC_BACKEND_URL"
$env:VITE_API_KEY="PASTE_THE_SAME_KEY_AS_BACKEND"
python tools/preflight_split_deploy.py
```

Expected output:
- health payload returned
- `/api/realtime/latest` returned
- metadata WebSocket connected
- `preflight: success`

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
- Bad RTSP URL in `streaming.jcfg`
- Browser cannot use raw RTSP (must use Janus/gateway URL)

CORS errors:
- Missing Vercel domain in `CORS_ORIGINS`.

