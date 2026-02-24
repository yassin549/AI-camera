# AIcam: Janus WebRTC + Metadata Overlay

AIcam now uses a media-gateway architecture:
- Browser video path: camera RTSP/H264 -> Janus WebRTC -> `<video>` (hardware decode).
- Python path: detection/tracking/identity only -> metadata WebSocket (`/api/realtime/ws`).
- Overlay path: metadata WS -> canvas draw over video.

Python frame streaming endpoints (`/api/media/mjpeg`, `/api/media/ws`) are still available as optional fallback but disabled by default.

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

2. Configure Janus mountpoint:
- Edit `janus/etc/janus/streaming.jcfg`.
- Set camera URL and password in `url = "rtsp://..."`.

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
- `VITE_WS_METADATA_URL` default: `ws://localhost:8080/ws/metadata`
- `VITE_JANUS_HTTP_URL` default: `http://localhost:8088/janus`
- `VITE_JANUS_MOUNTPOINT` default: `1`
- `VITE_VIDEO_WS_URL` default: `/api/media/ws`
- `VITE_DISABLE_WEBRTC=true` forces fallback chain (`WS-JPEG -> MJPEG`)

## Backend environment (optional)
- `AICAM_ENABLE_FRAME_STREAMING=1` enables Python frame endpoints (`/api/media/ws`, `/api/media/mjpeg`).
- Default is disabled to avoid server-side frame encode overhead.

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
- `GET /api/health` -> `{"status":"ok","time":"..."}`
- `GET /api/identities` -> identity list JSON
- `GET /api/realtime/latest` -> latest metadata payload
- `WS /api/realtime/ws` -> metadata stream

## Verification checklist
- Janus container is healthy: `docker compose ps janus`
- Janus player test works: open `janus/test-player.html` and see live video.
- Frontend Live page shows `Video: JANUS`.
- Boxes and IDs render in overlay from metadata WS.
- `curl http://127.0.0.1:8080/api/health` returns status JSON.
- `curl http://127.0.0.1:8080/api/identities` returns JSON list.
- Stop Janus: frontend should fail over to fallback transport instead of crashing.

## Debug checklist
- No Janus video: verify `VITE_JANUS_HTTP_URL` and `VITE_JANUS_MOUNTPOINT`.
- No overlay: verify metadata WS at `/api/realtime/ws`.
- High backend CPU: ensure `AICAM_ENABLE_FRAME_STREAMING` is not enabled.
- Camera pull issues in Janus: use `scripts/ffmpeg-push.sh` with `-c copy` and ingest via your gateway path.
