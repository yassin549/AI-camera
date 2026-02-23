# AIcam Premium UI

Production-ready React + TypeScript frontend for AIcam with:
- raw low-latency video transport via WebRTC (offer flow)
- automatic MJPEG fallback (`/stream.mjpeg`)
- client-side canvas overlays (boxes, labels, thumbs)
- realtime metadata via WebSocket (`/ws/metadata`)
- premium Live + Library UX with Framer Motion and Headless UI

## Stack
- React 18 + TypeScript (hooks)
- Vite
- Tailwind CSS
- Framer Motion
- Headless UI
- Jest + React Testing Library
- Node mock server (`/mock/server.js`)

## Project structure
- `src/main.tsx` app bootstrap + router + global providers
- `src/App.tsx` top bar + PremiumSwitch + route orchestration
- `src/pages/LiveView.tsx` video surface + overlays + roster + quick actions
- `src/pages/Library.tsx` identities grid, search, sort
- `src/pages/IdentityDetail.tsx` full-screen identity detail + actions
- `src/components/PremiumSwitch.tsx` accessible tab switch
- `src/components/VideoCanvas.tsx` WebRTC + fallback + canvas + click/focus
- `src/components/IdentityCard.tsx` compact library card
- `src/hooks/useRealtime.ts` WebSocket lifecycle + reconnection/backoff
- `src/config.ts` runtime endpoint + mock config
- `src/api/client.ts` typed API wrappers + payload normalization
- `src/styles/global.css` premium palette + glassmorphism tokens
- `src/__tests__/IdentityCard.test.tsx` smoke test for card -> detail route
- `mock/server.js` mock REST + WS metadata + MJPEG + WebRTC answer stub

## Prerequisites
- Node.js 18+ (Node 20 recommended)
- npm 9+

## Quick Start (Mock Backend)
1. Install dependencies:
```bash
npm i
```
2. Start the mock backend:
```bash
npm run mock
```
3. In a second terminal, start the frontend in mock mode:
```bash
npm run dev:mock
```
4. Open `http://localhost:5173`.

PowerShell example:
```powershell
npm i
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PWD'; npm run mock"
npm run dev:mock
```

## Run Against Real AIcam Backend
1. Ensure your backend serves:
   - `GET /api/identities`
   - `GET /api/identities/:id`
   - `POST /webrtc/offer`
   - `ws://.../ws/metadata` (or `wss://.../ws/metadata`)
2. Set environment overrides and run:
```powershell
$env:VITE_API_BASE='http://127.0.0.1:8080'
$env:VITE_WS_METADATA_URL='ws://127.0.0.1:8080/ws/metadata'
npm run dev
```

## Force MJPEG Fallback (Disable WebRTC)
```bash
VITE_DISABLE_WEBRTC=true npm run dev
```
PowerShell:
```powershell
$env:VITE_DISABLE_WEBRTC='true'; npm run dev
```

## Scripts
- `npm run dev` start Vite dev server (real backend defaults)
- `npm run dev:mock` start Vite dev server with mock-target proxy config
- `npm run build` type-check + production build
- `npm run test` run Jest/RTL tests
- `npm run mock` run local mock backend on `http://localhost:8787`

## Endpoint contracts
### REST
- `GET /api/identities`
- `GET /api/identities/:id`
- `POST /api/identities/:id/rename`
- `POST /api/identities/:id/merge`
- `DELETE /api/identities/:id`
- `POST /api/identities/:id/snapshot`
- `POST /api/identities/:id/mute`
- `POST /webrtc/offer`

### WebSocket
- `ws://<host>/ws/metadata` in dev
- `wss://<host>/ws/metadata` in production

Message shape:
```json
{
  "frame_id": 12345,
  "timestamp": "2026-02-22T14:01:10Z",
  "tracks": [
    {
      "track_id": 17,
      "bbox": [x, y, w, h],
      "identity_id": 3,
      "label": "ID:3 (0.82)",
      "modality": "face",
      "thumb": "/faces/3_last.jpg"
    }
  ]
}
```

## WebRTC notes
`/webrtc/offer` in the mock server is a signaling stub for local UI plumbing only. A real backend must:
1. accept SDP offers
2. generate valid SDP answers
3. attach a media source to `RTCPeerConnection`

If WebRTC fails, the UI falls back to `/stream.mjpeg` automatically.

## Why canvas overlays preserve quality
The `<video>` element displays the raw stream directly. Overlay annotations are drawn on a separate `<canvas>` layered above it. This avoids server-side re-encoding and keeps stream pixels unchanged while maintaining low-latency visual metadata.

## Pixel-perfect mode
LiveView includes a `Pixel-perfect` toggle. When enabled, the canvas backing store uses video resolution x DPR for sharper overlays on high-DPI displays.

## Large library scaling
For large identity sets, integrate virtualization (for example `@tanstack/react-virtual`) in `Library.tsx` grid rendering. The current implementation keeps lazy-loaded images and cheap card rendering as a baseline.

## Environment overrides
Optional env vars:
- `VITE_API_BASE` (default `http://localhost:8080`)
- `VITE_WS_METADATA_URL` (default `ws://localhost:8080/ws/metadata`)
- `VITE_WEBRTC_OFFER` (default `${VITE_API_BASE}/webrtc/offer`)
- `VITE_DISABLE_WEBRTC=true` (force MJPEG fallback)
- `VITE_USE_MOCK=1` (enable mock proxy mode in dev config)

Legacy compatibility:
- `REACT_APP_API_URL`
- `REACT_APP_WS_URL`
- `REACT_APP_WEBRTC_OFFER`
- `REACT_APP_MJPEG`
- `REACT_APP_USE_MOCK`

## Integration checklist for real backend
1. Implement `/webrtc/offer` with full signaling/media.
2. Emit metadata frames on `/ws/metadata` using the documented contract.
3. Expose identity REST endpoints.
4. Keep video raw; do not burn overlays server-side.
