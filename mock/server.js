import fs from "node:fs";
import path from "node:path";
import http from "node:http";
import { WebSocketServer, WebSocket } from "ws";

const PORT = Number(process.env.MOCK_PORT || 8787);
const HOST = process.env.MOCK_HOST || "0.0.0.0";
const ROOT = process.cwd();

const JPEG_FALLBACK_BASE64 =
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEhAQEBAPEA8PEA8PEA8PEA8QEA8QFREWFhURFRUYHSggGBolHRUVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGhAQGi0fHR0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAAEAAQMBIgACEQEDEQH/xAAXAAEBAQEAAAAAAAAAAAAAAAAAAQID/8QAFhEBAQEAAAAAAAAAAAAAAAAAAAER/9oADAMBAAIQAxAAAAG5hAAAAAAAAAAAAP/EABQQAQAAAAAAAAAAAAAAAAAAACD/2gAIAQEAAQUCF//EABQRAQAAAAAAAAAAAAAAAAAAACD/2gAIAQMBAT8BP//EABQRAQAAAAAAAAAAAAAAAAAAACD/2gAIAQIBAT8BP//Z";

function withCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
}

function sendJson(res, status, payload) {
  withCors(res);
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function parseJpegSize(buffer) {
  if (!buffer || buffer.length < 4 || buffer[0] !== 0xff || buffer[1] !== 0xd8) {
    return { width: 1280, height: 720 };
  }
  let offset = 2;
  while (offset + 9 < buffer.length) {
    if (buffer[offset] !== 0xff) {
      offset += 1;
      continue;
    }
    const marker = buffer[offset + 1];
    const length = buffer.readUInt16BE(offset + 2);
    const isSof =
      marker === 0xc0 || marker === 0xc1 || marker === 0xc2 || marker === 0xc3 || marker === 0xc9 || marker === 0xca;
    if (isSof) {
      return {
        height: buffer.readUInt16BE(offset + 5),
        width: buffer.readUInt16BE(offset + 7)
      };
    }
    offset += 2 + length;
  }
  return { width: 1280, height: 720 };
}

function loadFrameBuffers() {
  const candidates = [
    path.join(ROOT, "tmp_multiface.jpg"),
    path.join(ROOT, "tmp_multiface_hard.jpg"),
    path.join(ROOT, "tmp_blank.jpg")
  ];
  const buffers = candidates.filter((file) => fs.existsSync(file)).map((file) => fs.readFileSync(file));
  if (buffers.length > 0) {
    return buffers;
  }
  return [Buffer.from(JPEG_FALLBACK_BASE64, "base64")];
}

const frameBuffers = loadFrameBuffers();
const sourceSize = parseJpegSize(frameBuffers[0]);
const faceImageByName = new Map();

for (let i = 1; i <= 12; i += 1) {
  const key = `${i}_last.jpg`;
  faceImageByName.set(key, frameBuffers[(i - 1) % frameBuffers.length]);
}

let identities = Array.from({ length: 12 }).map((_, index) => {
  const id = String(index + 1);
  return {
    id,
    first_seen: new Date(Date.now() - (index + 18) * 1000 * 60).toISOString(),
    last_seen: new Date(Date.now() - index * 1000 * 25).toISOString(),
    face_samples: [`/faces/${id}_last.jpg`],
    body_samples: [`/faces/${((index + 1) % 12) + 1}_last.jpg`],
    stats: {
      frequency: 4 + ((index * 3) % 15)
    },
    timeline: [
      { at: new Date(Date.now() - (index + 20) * 1000 * 60).toISOString(), note: "First observed" },
      { at: new Date(Date.now() - (index + 2) * 1000 * 20).toISOString(), note: "Seen near front door" }
    ]
  };
});

function makeTrack(index, frameId) {
  const person = identities[index % identities.length];
  const width = 140 + ((index * 23) % 60);
  const height = 180 + ((index * 17) % 80);
  const maxX = Math.max(1, sourceSize.width - width - 12);
  const maxY = Math.max(1, sourceSize.height - height - 12);
  const x = ((frameId * (2 + index)) + index * 120) % maxX;
  const y = ((frameId * (1 + index)) + index * 60) % maxY;
  const score = (0.72 + ((frameId + index) % 18) / 100).toFixed(2);
  return {
    track_id: index + 1,
    bbox: [Math.round(x), Math.round(y), width, height],
    identity_id: Number(person.id),
    label: `ID:${person.id} (${score})`,
    modality: index % 2 === 0 ? "face" : "body",
    thumb: person.face_samples[0]
  };
}

const server = http.createServer((req, res) => {
  const method = req.method || "GET";
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

  if (method === "OPTIONS") {
    withCors(res);
    res.statusCode = 204;
    res.end();
    return;
  }

  if (method === "GET" && (url.pathname === "/mock/stream.mjpeg" || url.pathname === "/stream.mjpeg")) {
    withCors(res);
    res.writeHead(200, {
      "Cache-Control": "no-cache, no-store, must-revalidate",
      Connection: "close",
      Pragma: "no-cache",
      "Content-Type": "multipart/x-mixed-replace; boundary=frame"
    });
    let idx = 0;
    const timer = setInterval(() => {
      const frame = frameBuffers[idx % frameBuffers.length];
      idx += 1;
      res.write(`--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${frame.length}\r\n\r\n`);
      res.write(frame);
      res.write("\r\n");
    }, 120);
    req.on("close", () => clearInterval(timer));
    return;
  }

  if (method === "GET" && url.pathname.startsWith("/faces/")) {
    const name = decodeURIComponent(url.pathname.replace("/faces/", ""));
    const frame = faceImageByName.get(name) || frameBuffers[0];
    withCors(res);
    res.writeHead(200, { "Content-Type": "image/jpeg", "Cache-Control": "no-store" });
    res.end(frame);
    return;
  }

  if (method === "GET" && url.pathname === "/api/identities") {
    sendJson(res, 200, identities);
    return;
  }

  if (method === "GET" && /^\/api\/identities\/[^/]+$/.test(url.pathname)) {
    const id = decodeURIComponent(url.pathname.split("/").at(-1) || "");
    const identity = identities.find((item) => item.id === id);
    if (!identity) {
      sendJson(res, 404, { error: "Not found" });
      return;
    }
    sendJson(res, 200, {
      ...identity,
      samples: [...identity.face_samples, ...identity.body_samples]
    });
    return;
  }

  if (method === "DELETE" && /^\/api\/identities\/[^/]+$/.test(url.pathname)) {
    const id = decodeURIComponent(url.pathname.split("/").at(-1) || "");
    identities = identities.filter((item) => item.id !== id);
    sendJson(res, 200, { ok: true });
    return;
  }

  if (method === "POST" && (url.pathname.endsWith("/rename") || url.pathname.endsWith("/merge") || url.pathname.endsWith("/snapshot") || url.pathname.endsWith("/mute"))) {
    let body = "";
    req.on("data", (chunk) => {
      body += String(chunk);
    });
    req.on("end", () => {
      let parsed = {};
      try {
        parsed = body ? JSON.parse(body) : {};
      } catch {
        parsed = {};
      }
      if (url.pathname.endsWith("/rename")) {
        sendJson(res, 200, { ok: true, name: parsed.name || "" });
        return;
      }
      if (url.pathname.endsWith("/merge")) {
        sendJson(res, 200, { ok: true, merged_into: parsed.target_id || null });
        return;
      }
      sendJson(res, 200, { ok: true });
    });
    return;
  }

  if (method === "POST" && url.pathname === "/webrtc/offer") {
    let body = "";
    req.on("data", (chunk) => {
      body += String(chunk);
    });
    req.on("end", () => {
      let offer = {};
      try {
        offer = JSON.parse(body);
      } catch {
        offer = {};
      }
      // Stub answer only; real signaling/media transport must be implemented server-side.
      sendJson(res, 200, {
        type: "answer",
        sdp:
          offer?.sdp ||
          "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=AIcam mock answer\r\nt=0 0\r\na=inactive\r\n"
      });
    });
    return;
  }

  if (method === "GET" && url.pathname === "/healthz") {
    sendJson(res, 200, { ok: true, source: "mock" });
    return;
  }

  sendJson(res, 404, { error: "Not found" });
});

const ws = new WebSocketServer({ server, path: "/ws/metadata" });
let frameId = 1;

setInterval(() => {
  const tracks = [makeTrack(0, frameId), makeTrack(1, frameId), makeTrack(2, frameId)];
  const payload = JSON.stringify({
    frame_id: frameId,
    timestamp: new Date().toISOString(),
    tracks
  });
  frameId += 1;
  for (const client of ws.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}, 120);

server.listen(PORT, HOST, () => {
  // eslint-disable-next-line no-console
  console.log(`[mock] listening on http://${HOST}:${PORT}`);
  // eslint-disable-next-line no-console
  console.log(`[mock] stream: http://${HOST}:${PORT}/stream.mjpeg`);
  // eslint-disable-next-line no-console
  console.log(`[mock] metadata ws: ws://${HOST}:${PORT}/ws/metadata`);
});
