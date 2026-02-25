import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  postWebRtcOffer,
  resolveMetadataWsUrl,
  withApiKeyHeaders,
  withApiKeyQuery,
  type MetadataPayload,
  type TrackPayload
} from "../api/client";
import { API } from "../config";
import { useRealtime } from "../hooks/useRealtime";

interface VideoCanvasProps {
  className?: string;
  active?: boolean;
  pixelPerfect?: boolean;
  focusedTrackId?: number | null;
  metadataUrl?: string;
  janusHttpUrl?: string;
  janusMountpoint?: number;
  onTrackFocus?: (track: TrackPayload | null) => void;
  onRosterChange?: (tracks: TrackPayload[]) => void;
}

type TransportMode = "connecting" | "janus" | "webrtc" | "mjpeg" | "error";

interface ProjectedTrack {
  track: TrackPayload;
  x: number;
  y: number;
  w: number;
  h: number;
}

interface CanvasMetrics {
  cssWidth: number;
  cssHeight: number;
  sourceWidth: number;
  sourceHeight: number;
}

interface DrawCache {
  tracksKey: string;
  sizeKey: string;
  focusedTrackId: number | null;
}

interface RosterEmitCache {
  key: string;
  at: number;
}

const THUMB_SIZE = 34;

function drawLabel(ctx: CanvasRenderingContext2D, label: string, x: number, y: number): void {
  ctx.font = "12px system-ui, -apple-system, Segoe UI, sans-serif";
  const metrics = ctx.measureText(label);
  const width = Math.ceil(metrics.width + 18);
  const height = 22;
  const radius = 11;

  const pillX = x;
  const pillY = Math.max(4, y - height - 8);
  ctx.fillStyle = "rgba(14, 22, 36, 0.9)";
  ctx.beginPath();
  ctx.moveTo(pillX + radius, pillY);
  ctx.lineTo(pillX + width - radius, pillY);
  ctx.quadraticCurveTo(pillX + width, pillY, pillX + width, pillY + radius);
  ctx.lineTo(pillX + width, pillY + height - radius);
  ctx.quadraticCurveTo(pillX + width, pillY + height, pillX + width - radius, pillY + height);
  ctx.lineTo(pillX + radius, pillY + height);
  ctx.quadraticCurveTo(pillX, pillY + height, pillX, pillY + height - radius);
  ctx.lineTo(pillX, pillY + radius);
  ctx.quadraticCurveTo(pillX, pillY, pillX + radius, pillY);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "rgba(229, 233, 255, 0.95)";
  ctx.fillText(label, pillX + 9, pillY + 15);
}

function getOrCreateImage(url: string, cache: Map<string, HTMLImageElement>): HTMLImageElement {
  const existing = cache.get(url);
  if (existing) {
    return existing;
  }
  const img = new Image();
  img.decoding = "async";
  img.loading = "eager";
  img.src = url;
  cache.set(url, img);
  return img;
}

function resolveApiUrl(pathOrUrl: string): string {
  if (!pathOrUrl) {
    return pathOrUrl;
  }
  if (/^(https?:)?\/\//i.test(pathOrUrl) || pathOrUrl.startsWith("blob:") || pathOrUrl.startsWith("data:")) {
    return pathOrUrl;
  }
  if (!API.REST_BASE) {
    return pathOrUrl;
  }
  const base = API.REST_BASE.replace(/\/+$/, "");
  const path = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${base}${path}`;
}

function nextTx(): string {
  return Math.random().toString(36).slice(2, 12);
}

function computeTracksKey(payload: MetadataPayload | null): string {
  if (!payload || payload.tracks.length === 0) {
    return "none";
  }
  return payload.tracks
    .map((track) => {
      const [x, y, w, h] = track.bbox;
      return `${track.track_id}:${x},${y},${w},${h}:${track.identity_id ?? "n"}:${track.label}`;
    })
    .join("|");
}

function computeRosterKey(tracks: TrackPayload[]): string {
  if (!tracks.length) {
    return "none";
  }
  return tracks
    .map((track) => `${track.track_id}:${track.identity_id ?? "n"}:${track.label}`)
    .join("|");
}

async function waitForIceGatheringComplete(pc: RTCPeerConnection, timeoutMs = 2000): Promise<void> {
  if (pc.iceGatheringState === "complete") {
    return;
  }
  await new Promise<void>((resolve) => {
    const timeout = window.setTimeout(() => {
      pc.removeEventListener("icegatheringstatechange", onStateChange);
      resolve();
    }, timeoutMs);
    const onStateChange = () => {
      if (pc.iceGatheringState === "complete") {
        window.clearTimeout(timeout);
        pc.removeEventListener("icegatheringstatechange", onStateChange);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", onStateChange);
  });
}

export function VideoCanvas({
  className,
  active = true,
  pixelPerfect = true,
  focusedTrackId = null,
  metadataUrl,
  janusHttpUrl = API.JANUS_HTTP,
  janusMountpoint = API.JANUS_MOUNTPOINT,
  onTrackFocus,
  onRosterChange
}: VideoCanvasProps): JSX.Element {
  const resolvedWsUrl = useMemo(
    () => withApiKeyQuery(metadataUrl ?? resolveMetadataWsUrl()),
    [metadataUrl]
  );
  const resolvedJanusHttpUrl = useMemo(() => resolveApiUrl(janusHttpUrl).replace(/\/+$/, ""), [janusHttpUrl]);
  const resolvedMjpegUrl = useMemo(() => withApiKeyQuery(resolveApiUrl("/api/media/mjpeg")), []);
  const resolvedJanusMountpoint = useMemo(
    () => (Number.isFinite(janusMountpoint) && Number(janusMountpoint) > 0 ? Number(janusMountpoint) : 1),
    [janusMountpoint]
  );
  const disableWebRtc = (import.meta.env.VITE_DISABLE_WEBRTC as string | undefined) === "true";
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mjpegRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const projectedTracksRef = useRef<ProjectedTrack[]>([]);
  const thumbCacheRef = useRef<Map<string, HTMLImageElement>>(new Map());
  const latestMetadataRef = useRef<MetadataPayload | null>(null);
  const drawCacheRef = useRef<DrawCache>({ tracksKey: "", sizeKey: "", focusedTrackId: null });
  const rosterEmitRef = useRef<RosterEmitCache>({ key: "", at: 0 });
  const metricsRef = useRef<CanvasMetrics>({ cssWidth: 0, cssHeight: 0, sourceWidth: 0, sourceHeight: 0 });
  const resizeKeyRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const janusPcRef = useRef<RTCPeerConnection | null>(null);
  const janusAbortRef = useRef<AbortController | null>(null);
  const janusSessionRef = useRef<number | null>(null);
  const janusHandleRef = useRef<number | null>(null);
  const janusKeepaliveRef = useRef<number | null>(null);
  const transportModeRef = useRef<TransportMode>("connecting");

  const [transportMode, setTransportMode] = useState<TransportMode>("connecting");
  const [transportError, setTransportError] = useState<string | null>(null);
  const [faceDetectorUnavailable, setFaceDetectorUnavailable] = useState(false);

  const realtime = useRealtime({
    url: resolvedWsUrl,
    enabled: active,
    onMessage: (payload) => {
      const resolvedTracks = payload.tracks.map((track) => ({
        ...track,
        thumb: track.thumb ? resolveApiUrl(track.thumb) : undefined
      }));
      const resolvedPayload: MetadataPayload = {
        ...payload,
        tracks: resolvedTracks
      };
      latestMetadataRef.current = resolvedPayload;
      if (onRosterChange) {
        const nextKey = computeRosterKey(resolvedTracks);
        const now = performance.now();
        const prev = rosterEmitRef.current;
        if (nextKey !== prev.key || now - prev.at >= 200) {
          rosterEmitRef.current = { key: nextKey, at: now };
          onRosterChange(resolvedTracks);
        }
      }
    }
  });

  const closeTransport = () => {
    if (janusKeepaliveRef.current !== null) {
      window.clearInterval(janusKeepaliveRef.current);
      janusKeepaliveRef.current = null;
    }
    const abort = janusAbortRef.current;
    if (abort) {
      abort.abort();
      janusAbortRef.current = null;
    }
    const pc = janusPcRef.current;
    if (!pc) {
      // continue: session may still need cleanup.
    } else {
      pc.ontrack = null;
      pc.onicecandidate = null;
      pc.close();
      janusPcRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.removeAttribute("src");
      video.style.display = "block";
    }
    const mjpeg = mjpegRef.current;
    if (mjpeg) {
      mjpeg.onload = null;
      mjpeg.onerror = null;
      mjpeg.style.display = "none";
      mjpeg.removeAttribute("src");
    }

    const sessionId = janusSessionRef.current;
    janusSessionRef.current = null;
    janusHandleRef.current = null;
    if (!sessionId) {
      return;
    }
    const payload = {
      janus: "destroy",
      transaction: nextTx(),
    };
    void fetch(`${resolvedJanusHttpUrl}/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => undefined);
  };

  useEffect(() => {
    let cancelled = false;
    const video = videoRef.current;
    const mjpeg = mjpegRef.current;
    if (!active) {
      closeTransport();
      transportModeRef.current = "connecting";
      setTransportMode("connecting");
      setTransportError(null);
      return;
    }
    if (!video || !mjpeg) {
      return;
    }

    const hideMjpeg = () => {
      mjpeg.style.display = "none";
      mjpeg.removeAttribute("src");
    };
    const showVideo = () => {
      video.style.display = "block";
      hideMjpeg();
    };
    const showMjpeg = (url: string) => {
      video.pause();
      video.srcObject = null;
      video.removeAttribute("src");
      video.style.display = "none";
      mjpeg.style.display = "block";
      mjpeg.src = url;
    };

    const janusPost = async (
      path: string,
      body: Record<string, unknown>,
      signal: AbortSignal
    ): Promise<Record<string, unknown>> => {
      const response = await fetch(`${resolvedJanusHttpUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, transaction: nextTx() }),
        signal,
      });
      if (!response.ok) {
        throw new Error(`Janus HTTP ${response.status}`);
      }
      const payload = (await response.json()) as Record<string, unknown>;
      if (payload.janus === "error") {
        const error = payload.error as { code?: number; reason?: string } | undefined;
        const reason = error?.reason ?? `code ${error?.code ?? "unknown"}`;
        throw new Error(`Janus error: ${reason}`);
      }
      return payload;
    };

    const startJanusTransport = async (): Promise<{ ok: boolean; error: string }> => {
      if (!resolvedJanusHttpUrl || typeof RTCPeerConnection === "undefined") {
        return { ok: false, error: "Janus endpoint or browser WebRTC unavailable" };
      }
      const abort = new AbortController();
      const signal = abort.signal;
      janusAbortRef.current = abort;
      let sessionId: number | null = null;
      let handleId: number | null = null;
      try {
        const created = await janusPost("", { janus: "create" }, signal);
        sessionId = Number((created.data as { id?: number } | undefined)?.id ?? -1);
        if (!Number.isFinite(sessionId) || sessionId <= 0) {
          throw new Error("Janus create session failed");
        }

        const attached = await janusPost(
          `/${sessionId}`,
          { janus: "attach", plugin: "janus.plugin.streaming" },
          signal
        );
        handleId = Number((attached.data as { id?: number } | undefined)?.id ?? -1);
        if (!Number.isFinite(handleId) || handleId <= 0) {
          throw new Error("Janus attach plugin failed");
        }

        janusSessionRef.current = sessionId;
        janusHandleRef.current = handleId;
        const pc = new RTCPeerConnection();
        janusPcRef.current = pc;
        pc.ontrack = (event) => {
          if (event.streams[0]) {
            showVideo();
            video.srcObject = event.streams[0];
            video.play().catch(() => undefined);
            transportModeRef.current = "janus";
            setTransportMode("janus");
            setTransportError(null);
          }
        };
        pc.onicecandidate = (event) => {
          const sid = janusSessionRef.current;
          const hid = janusHandleRef.current;
          if (!sid || !hid || signal.aborted) {
            return;
          }
          void janusPost(
            `/${sid}/${hid}`,
            {
              janus: "trickle",
              candidate: event.candidate ?? { completed: true },
            },
            signal
          ).catch(() => undefined);
        };

        await janusPost(
          `/${sessionId}/${handleId}`,
          { janus: "message", body: { request: "watch", id: resolvedJanusMountpoint } },
          signal
        );

        janusKeepaliveRef.current = window.setInterval(() => {
          if (signal.aborted || !janusSessionRef.current) {
            return;
          }
          void janusPost(
            `/${janusSessionRef.current}`,
            { janus: "keepalive" },
            signal
          ).catch(() => undefined);
        }, 25000);

        const pollEvents = async () => {
          while (!signal.aborted && janusSessionRef.current && janusHandleRef.current) {
            const rid = Date.now();
            const response = await fetch(
              `${resolvedJanusHttpUrl}/${janusSessionRef.current}?rid=${rid}&maxev=10`,
              { signal }
            );
            if (!response.ok) {
              throw new Error(`Janus poll ${response.status}`);
            }
            const event = (await response.json()) as Record<string, unknown>;
            const eventSender = Number(event.sender ?? -1);

            if (
              event.janus === "event" &&
              eventSender === janusHandleRef.current &&
              event.jsep &&
              typeof event.jsep === "object" &&
              janusPcRef.current
            ) {
              const remote = event.jsep as RTCSessionDescriptionInit;
              await janusPcRef.current.setRemoteDescription(remote);
              const answer = await janusPcRef.current.createAnswer();
              await janusPcRef.current.setLocalDescription(answer);
              await janusPost(
                `/${janusSessionRef.current}/${janusHandleRef.current}`,
                {
                  janus: "message",
                  body: { request: "start" },
                  jsep: janusPcRef.current.localDescription,
                },
                signal
              );
            }

            if (
              event.janus === "trickle" &&
              eventSender === janusHandleRef.current &&
              event.candidate &&
              janusPcRef.current
            ) {
              const candidate = event.candidate as { completed?: boolean; candidate?: string };
              try {
                await janusPcRef.current.addIceCandidate(candidate.completed ? null : (candidate as RTCIceCandidateInit));
              } catch {
                // Ignore invalid trickle candidates and continue polling.
              }
            }
          }
        };

        void pollEvents().catch((err: unknown) => {
          if (signal.aborted || cancelled) {
            return;
          }
          const message = err instanceof Error ? err.message : "Janus polling failed";
          setTransportError(`Janus stream error: ${message}`);
          transportModeRef.current = "error";
          setTransportMode("error");
          closeTransport();
        });

        return { ok: true, error: "" };
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Janus negotiation failed";
        if (sessionId && !signal.aborted) {
          void janusPost(`/${sessionId}`, { janus: "destroy" }, signal).catch(() => undefined);
        }
        closeTransport();
        return { ok: false, error: message };
      }
    };

    const startApiWebRtcTransport = async (): Promise<{ ok: boolean; error: string }> => {
      if (typeof RTCPeerConnection === "undefined") {
        return { ok: false, error: "Browser WebRTC unavailable" };
      }
      try {
        const pc = new RTCPeerConnection();
        janusPcRef.current = pc;
        const firstTrack = new Promise<boolean>((resolve) => {
          const timeout = window.setTimeout(() => resolve(false), 7000);
          pc.ontrack = (event) => {
            if (!event.streams[0]) {
              return;
            }
            window.clearTimeout(timeout);
            showVideo();
            video.srcObject = event.streams[0];
            video.play().catch(() => undefined);
            transportModeRef.current = "webrtc";
            setTransportMode("webrtc");
            setTransportError(null);
            resolve(true);
          };
        });

        pc.addTransceiver("video", { direction: "recvonly" });
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGatheringComplete(pc, 2000);
        const local = pc.localDescription;
        if (!local?.sdp) {
          throw new Error("Local WebRTC offer is empty");
        }
        const answer = await postWebRtcOffer({ sdp: local.sdp, type: "offer" });
        await pc.setRemoteDescription(answer as RTCSessionDescriptionInit);
        const gotTrack = await firstTrack;
        if (!gotTrack) {
          throw new Error("No media track received");
        }
        return { ok: true, error: "" };
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "API WebRTC negotiation failed";
        closeTransport();
        return { ok: false, error: message };
      }
    };

    const startMjpegTransport = async (): Promise<{ ok: boolean; error: string }> => {
      const separator = resolvedMjpegUrl.includes("?") ? "&" : "?";
      const probeUrl = `${resolvedMjpegUrl}${separator}probe=1&_=${Date.now()}`;
      const probeAbort = new AbortController();
      const timeoutId = window.setTimeout(() => probeAbort.abort(), 3000);
      try {
        const response = await fetch(probeUrl, {
          method: "GET",
          headers: withApiKeyHeaders(),
          signal: probeAbort.signal
        });
        window.clearTimeout(timeoutId);
        if (!response.ok) {
          let details = "";
          try {
            details = await response.text();
          } catch {
            details = "";
          }
          return {
            ok: false,
            error: details ? `HTTP ${response.status}: ${details}` : `HTTP ${response.status}`,
          };
        }
        void response.body?.cancel().catch(() => undefined);
      } catch (err: unknown) {
        window.clearTimeout(timeoutId);
        const message = err instanceof Error ? err.message : "MJPEG probe failed";
        return { ok: false, error: message };
      }

      const streamUrl = `${resolvedMjpegUrl}${separator}stream=1&_=${Date.now()}`;
      const loaded = await new Promise<boolean>((resolve) => {
        const timeout = window.setTimeout(() => {
          mjpeg.onload = null;
          mjpeg.onerror = null;
          resolve(false);
        }, 5000);
        mjpeg.onload = () => {
          window.clearTimeout(timeout);
          mjpeg.onload = null;
          mjpeg.onerror = null;
          resolve(true);
        };
        mjpeg.onerror = () => {
          window.clearTimeout(timeout);
          mjpeg.onload = null;
          mjpeg.onerror = null;
          resolve(false);
        };
        showMjpeg(streamUrl);
      });

      if (!loaded) {
        hideMjpeg();
        return { ok: false, error: "MJPEG stream did not start" };
      }
      transportModeRef.current = "mjpeg";
      setTransportMode("mjpeg");
      setTransportError(null);
      return { ok: true, error: "" };
    };

    const bootstrapVideo = async () => {
      setTransportError(null);
      transportModeRef.current = "connecting";
      setTransportMode("connecting");

      const failures: string[] = [];
      if (!disableWebRtc) {
        const janusReady = await startJanusTransport();
        if (cancelled) {
          return;
        }
        if (janusReady.ok) {
          return;
        }
        failures.push(`Janus: ${janusReady.error}`);

        const apiWebRtcReady = await startApiWebRtcTransport();
        if (cancelled) {
          return;
        }
        if (apiWebRtcReady.ok) {
          return;
        }
        failures.push(`API WebRTC: ${apiWebRtcReady.error}`);
      } else {
        failures.push("WebRTC disabled by VITE_DISABLE_WEBRTC=true");
      }

      const mjpegReady = await startMjpegTransport();
      if (cancelled) {
        return;
      }
      if (mjpegReady.ok) {
        return;
      }
      failures.push(`MJPEG: ${mjpegReady.error}`);
      transportModeRef.current = "error";
      setTransportMode("error");
      setTransportError(failures.join(" | "));
    };

    bootstrapVideo();

    return () => {
      cancelled = true;
      closeTransport();
    };
  }, [active, disableWebRtc, resolvedJanusHttpUrl, resolvedJanusMountpoint, resolvedMjpegUrl]);

  useEffect(() => {
    if (!active) {
      setFaceDetectorUnavailable(false);
      return;
    }
    let cancelled = false;
    const healthUrl = resolveApiUrl("/api/health");

    const pollHealth = async () => {
      try {
        const response = await fetch(healthUrl, { method: "GET", headers: withApiKeyHeaders() });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as { face_detector_available?: boolean };
        if (!cancelled) {
          setFaceDetectorUnavailable(payload.face_detector_available === false);
        }
      } catch {
        // Health polling is best-effort; keep current state on transient failures.
      }
    };

    void pollHealth();
    const timer = window.setInterval(() => {
      void pollHealth();
    }, 10000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [active]);

  useEffect(() => {
    if (!active) {
      latestMetadataRef.current = null;
      drawCacheRef.current = { tracksKey: "", sizeKey: "", focusedTrackId: null };
      rosterEmitRef.current = { key: "", at: 0 };
      onTrackFocus?.(null);
      onRosterChange?.([]);
    }
  }, [active, onTrackFocus, onRosterChange]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const observer = new ResizeObserver(() => {
      resizeKeyRef.current += 1;
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const container = containerRef.current;
      if (!canvas || !container) {
        rafRef.current = window.requestAnimationFrame(draw);
        return;
      }

      // Determine source dimensions from the active video element
      let nativeW = 0;
      let nativeH = 0;
      if (video) {
        nativeW = video.videoWidth;
        nativeH = video.videoHeight;
      }
      if ((!nativeW || !nativeH) && mjpegRef.current) {
        nativeW = mjpegRef.current.naturalWidth;
        nativeH = mjpegRef.current.naturalHeight;
      }

      const rect = container.getBoundingClientRect();
      const cssWidth = Math.max(1, Math.round(rect.width));
      const cssHeight = Math.max(1, Math.round(rect.height));
      const latest = latestMetadataRef.current;
      const metadataSourceWidth = Number(latest?.source_width ?? 0);
      const metadataSourceHeight = Number(latest?.source_height ?? 0);
      const sourceWidth = Math.max(1, Math.round(metadataSourceWidth > 0 ? metadataSourceWidth : nativeW || cssWidth));
      const sourceHeight = Math.max(1, Math.round(metadataSourceHeight > 0 ? metadataSourceHeight : nativeH || cssHeight));
      const dpr = window.devicePixelRatio || 1;

      const bufferWidth = pixelPerfect ? Math.round(sourceWidth * dpr) : Math.round(cssWidth * dpr);
      const bufferHeight = pixelPerfect ? Math.round(sourceHeight * dpr) : Math.round(cssHeight * dpr);

      if (
        canvas.width !== bufferWidth ||
        canvas.height !== bufferHeight ||
        canvas.style.width !== `${cssWidth}px` ||
        canvas.style.height !== `${cssHeight}px`
      ) {
        canvas.width = bufferWidth;
        canvas.height = bufferHeight;
        canvas.style.width = `${cssWidth}px`;
        canvas.style.height = `${cssHeight}px`;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        rafRef.current = window.requestAnimationFrame(draw);
        return;
      }
      ctx.setTransform(canvas.width / cssWidth, 0, 0, canvas.height / cssHeight, 0, 0);
      ctx.imageSmoothingEnabled = true;

      metricsRef.current = {
        cssWidth,
        cssHeight,
        sourceWidth,
        sourceHeight
      };

      const tracksKey = computeTracksKey(latest);
      const sizeKey = `${cssWidth}x${cssHeight}:${sourceWidth}x${sourceHeight}:${resizeKeyRef.current}:${pixelPerfect ? "pp" : "fit"}`;
      const cache = drawCacheRef.current;
      const shouldDraw =
        cache.tracksKey !== tracksKey ||
        cache.sizeKey !== sizeKey ||
        cache.focusedTrackId !== focusedTrackId;
      if (!shouldDraw || !active) {
        rafRef.current = window.requestAnimationFrame(draw);
        return;
      }

      cache.tracksKey = tracksKey;
      cache.sizeKey = sizeKey;
      cache.focusedTrackId = focusedTrackId;

      ctx.clearRect(0, 0, cssWidth, cssHeight);
      projectedTracksRef.current = [];

      if (!latest || sourceWidth < 2 || sourceHeight < 2) {
        rafRef.current = window.requestAnimationFrame(draw);
        return;
      }

      const fitScale = Math.min(cssWidth / sourceWidth, cssHeight / sourceHeight);
      const renderWidth = sourceWidth * fitScale;
      const renderHeight = sourceHeight * fitScale;
      const offsetX = (cssWidth - renderWidth) / 2;
      const offsetY = (cssHeight - renderHeight) / 2;
      const scaleX = renderWidth / sourceWidth;
      const scaleY = renderHeight / sourceHeight;

      for (const track of latest.tracks) {
        const [x1, y1, x2, y2] = track.bbox;
        const bw = x2 - x1;
        const bh = y2 - y1;
        if (bw <= 0 || bh <= 0) {
          console.error("Invalid bbox(xyxy) received from metadata stream", { bbox: track.bbox, trackId: track.track_id });
          continue;
        }
        const bx = x1;
        const by = y1;
        const x = offsetX + bx * scaleX;
        const y = offsetY + by * scaleY;
        const w = bw * scaleX;
        const h = bh * scaleY;
        if (w <= 1 || h <= 1) {
          continue;
        }

        const isFocused = focusedTrackId !== null && focusedTrackId === track.track_id;
        const ageRatio = Math.max(0, Math.min(1, Number(track.age_ratio ?? 0)));
        const freshness = 1 - ageRatio;
        const strokeAlpha = isFocused ? 1 : 0.35 + freshness * 0.6;
        const fillAlpha = isFocused ? 0.13 : 0.05 + freshness * 0.11;
        const accent = isFocused
          ? "rgba(255, 198, 89, 1)"
          : `rgba(124, 92, 255, ${Math.max(0.2, Math.min(1, strokeAlpha)).toFixed(3)})`;
        const fill = isFocused
          ? "rgba(255, 198, 89, 0.13)"
          : `rgba(124, 92, 255, ${Math.max(0.03, Math.min(0.25, fillAlpha)).toFixed(3)})`;

        ctx.fillStyle = fill;
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.3;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);

        if (track.thumb) {
          const image = getOrCreateImage(track.thumb, thumbCacheRef.current);
          if (image.complete) {
            ctx.drawImage(image, x + 6, y + 6, THUMB_SIZE, THUMB_SIZE);
            ctx.strokeStyle = "rgba(255,255,255,0.7)";
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 6, y + 6, THUMB_SIZE, THUMB_SIZE);
          }
        }

        drawLabel(ctx, track.label, x, y);
        projectedTracksRef.current.push({ track, x, y, w, h });
      }

      rafRef.current = window.requestAnimationFrame(draw);
    };

    rafRef.current = window.requestAnimationFrame(draw);
    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
      }
    };
  }, [active, focusedTrackId, pixelPerfect]);

  const handleCanvasClick = (event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const matches = [...projectedTracksRef.current].reverse();
    const hit = matches.find((track) => x >= track.x && x <= track.x + track.w && y >= track.y && y <= track.y + track.h);
    if (!hit) {
      onTrackFocus?.(null);
      return;
    }
    onTrackFocus?.(hit.track);
  };

  return (
    <div className={`relative h-full w-full overflow-hidden rounded-2xl bg-black ${className ?? ""}`} ref={containerRef}>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="h-full w-full object-contain"
        style={{ display: "block" }}
        aria-label="Live camera stream"
      />
      <img
        ref={mjpegRef}
        className="absolute inset-0 h-full w-full object-contain"
        style={{ display: "none" }}
        alt="Live camera MJPEG stream"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 cursor-pointer"
        onClick={handleCanvasClick}
        role="application"
        aria-label="Identity overlays"
      />
      <div className="pointer-events-none absolute bottom-3 left-3 flex items-center gap-2 rounded-lg bg-slate-900/70 px-3 py-1.5 text-xs text-slate-100 backdrop-blur-md">
        <span className="h-2 w-2 rounded-full bg-emerald-400" />
        <span>Video: {transportMode.toUpperCase()}</span>
        <span>Metadata: {realtime.status.toUpperCase()}</span>
      </div>
      {transportError ? (
        <div className="pointer-events-none absolute right-3 top-3 rounded-lg bg-amber-300/90 px-3 py-2 text-xs font-medium text-slate-950 shadow-soft">
          Live stream unavailable ({transportError})
        </div>
      ) : null}
      {faceDetectorUnavailable ? (
        <div className="pointer-events-none absolute right-3 top-14 rounded-lg bg-rose-400/90 px-3 py-2 text-xs font-medium text-slate-950 shadow-soft">
          Face detector unavailable (mediapipe not installed)
        </div>
      ) : null}
      {realtime.error ? (
        <div className="pointer-events-none absolute right-3 top-24 rounded-lg bg-rose-400/90 px-3 py-2 text-xs font-medium text-slate-950 shadow-soft">
          Metadata channel reconnecting...
        </div>
      ) : null}
    </div>
  );
}
