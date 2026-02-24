import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  resolveMetadataWsUrl,
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
  videoWsUrl?: string;
  fallbackSrc?: string;
  onTrackFocus?: (track: TrackPayload | null) => void;
  onRosterChange?: (tracks: TrackPayload[]) => void;
}

type TransportMode = "connecting" | "janus" | "wsjpeg" | "mjpeg";

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
  frameId: number;
  sizeKey: string;
  focusedTrackId: number | null;
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

function resolveWsUrl(pathOrUrl: string): string {
  if (!pathOrUrl) {
    return "";
  }
  if (/^wss?:\/\//i.test(pathOrUrl)) {
    return pathOrUrl;
  }
  if (/^https?:\/\//i.test(pathOrUrl)) {
    return pathOrUrl.replace(/^http/i, "ws");
  }
  if (pathOrUrl.startsWith("/")) {
    if (API.REST_BASE && /^https?:\/\//i.test(API.REST_BASE)) {
      return `${API.REST_BASE.replace(/^http/i, "ws").replace(/\/+$/, "")}${pathOrUrl}`;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}${pathOrUrl}`;
  }
  return pathOrUrl;
}

function nextTx(): string {
  return Math.random().toString(36).slice(2, 12);
}

export function VideoCanvas({
  className,
  active = true,
  pixelPerfect = true,
  focusedTrackId = null,
  metadataUrl,
  janusHttpUrl = API.JANUS_HTTP,
  janusMountpoint = API.JANUS_MOUNTPOINT,
  videoWsUrl = API.VIDEO_WS,
  fallbackSrc = API.MJPEG_FALLBACK,
  onTrackFocus,
  onRosterChange
}: VideoCanvasProps): JSX.Element {
  const resolvedWsUrl = useMemo(() => metadataUrl ?? resolveMetadataWsUrl(), [metadataUrl]);
  const resolvedJanusHttpUrl = useMemo(() => resolveApiUrl(janusHttpUrl).replace(/\/+$/, ""), [janusHttpUrl]);
  const resolvedJanusMountpoint = useMemo(
    () => (Number.isFinite(janusMountpoint) && Number(janusMountpoint) > 0 ? Number(janusMountpoint) : 1),
    [janusMountpoint]
  );
  const resolvedVideoWsUrl = useMemo(() => resolveWsUrl(videoWsUrl), [videoWsUrl]);
  const resolvedFallbackSrc = useMemo(() => resolveApiUrl(fallbackSrc), [fallbackSrc]);
  const disableWebRtc = (import.meta.env.VITE_DISABLE_WEBRTC as string | undefined) === "true";
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mjpegImgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const projectedTracksRef = useRef<ProjectedTrack[]>([]);
  const thumbCacheRef = useRef<Map<string, HTMLImageElement>>(new Map());
  const latestMetadataRef = useRef<MetadataPayload | null>(null);
  const drawCacheRef = useRef<DrawCache>({ frameId: -1, sizeKey: "", focusedTrackId: null });
  const metricsRef = useRef<CanvasMetrics>({ cssWidth: 0, cssHeight: 0, sourceWidth: 0, sourceHeight: 0 });
  const resizeKeyRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const janusPcRef = useRef<RTCPeerConnection | null>(null);
  const janusAbortRef = useRef<AbortController | null>(null);
  const janusSessionRef = useRef<number | null>(null);
  const janusHandleRef = useRef<number | null>(null);
  const janusKeepaliveRef = useRef<number | null>(null);
  const wsVideoRef = useRef<WebSocket | null>(null);
  const wsBlobUrlRef = useRef<string | null>(null);
  const transportModeRef = useRef<TransportMode>("connecting");

  const [transportMode, setTransportMode] = useState<TransportMode>("connecting");
  const [transportError, setTransportError] = useState<string | null>(null);

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
      onRosterChange?.(resolvedTracks);
    }
  });

  const closeJanusTransport = () => {
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

  const releaseWsBlobUrl = () => {
    const url = wsBlobUrlRef.current;
    if (!url) {
      return;
    }
    window.URL.revokeObjectURL(url);
    wsBlobUrlRef.current = null;
  };

  const closeWsVideo = () => {
    const ws = wsVideoRef.current;
    if (ws) {
      ws.onopen = null;
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.close();
      wsVideoRef.current = null;
    }
    releaseWsBlobUrl();
  };

  const activateMjpegFallback = () => {
    const video = videoRef.current;
    if (video) {
      video.srcObject = null;
      video.removeAttribute("src");
      video.style.display = "none";
    }
    closeJanusTransport();
    closeWsVideo();
    const img = mjpegImgRef.current;
    if (img) {
      img.src = resolvedFallbackSrc;
      img.style.display = "block";
    }
    transportModeRef.current = "mjpeg";
    setTransportMode("mjpeg");
  };

  const activateWsJpegFallback = async (): Promise<boolean> => {
    if (!resolvedVideoWsUrl) {
      return false;
    }
    const img = mjpegImgRef.current;
    if (!img) {
      return false;
    }
    closeWsVideo();
    closeJanusTransport();
    img.style.display = "block";

    return await new Promise<boolean>((resolve) => {
      let settled = false;
      const finish = (ok: boolean) => {
        if (settled) {
          return;
        }
        settled = true;
        resolve(ok);
      };

      try {
        const ws = new WebSocket(resolvedVideoWsUrl);
        ws.binaryType = "arraybuffer";
        wsVideoRef.current = ws;
        const timeout = window.setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            ws.close();
            finish(false);
          }
        }, 1500);

        ws.onopen = () => {
          window.clearTimeout(timeout);
          transportModeRef.current = "wsjpeg";
          setTransportMode("wsjpeg");
          finish(true);
        };

        ws.onmessage = (event) => {
          const payload =
            event.data instanceof Blob
              ? event.data
              : new Blob([event.data as ArrayBuffer], { type: "image/jpeg" });
          const nextUrl = window.URL.createObjectURL(payload);
          const previous = wsBlobUrlRef.current;
          wsBlobUrlRef.current = nextUrl;
          img.src = nextUrl;
          if (previous) {
            window.setTimeout(() => window.URL.revokeObjectURL(previous), 1200);
          }
        };

        ws.onerror = () => {
          window.clearTimeout(timeout);
          if (ws.readyState !== WebSocket.OPEN) {
            finish(false);
          }
        };

        ws.onclose = () => {
          window.clearTimeout(timeout);
          if (wsVideoRef.current === ws) {
            wsVideoRef.current = null;
          }
          if (!settled) {
            finish(false);
            return;
          }
          if (transportModeRef.current === "wsjpeg") {
            activateMjpegFallback();
          }
        };
      } catch {
        finish(false);
      }
    });
  };

  useEffect(() => {
    let cancelled = false;
    const video = videoRef.current;
    const img = mjpegImgRef.current;
    if (!active) {
      closeJanusTransport();
      closeWsVideo();
      if (video) {
        video.pause();
        video.srcObject = null;
        video.removeAttribute("src");
      }
      if (img) {
        img.removeAttribute("src");
        img.style.display = "none";
      }
      transportModeRef.current = "connecting";
      setTransportMode("connecting");
      return;
    }
    if (!video) {
      return;
    }

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
      return (await response.json()) as Record<string, unknown>;
    };

    const startJanusTransport = async (): Promise<boolean> => {
      if (!resolvedJanusHttpUrl || typeof RTCPeerConnection === "undefined") {
        return false;
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
            closeWsVideo();
            if (img) {
              img.removeAttribute("src");
              img.style.display = "none";
            }
            video.style.display = "block";
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
              `${resolvedJanusHttpUrl}/${janusSessionRef.current}?rid=${rid}&maxev=1`,
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

        void pollEvents().catch(async (err: unknown) => {
          if (signal.aborted || cancelled) {
            return;
          }
          const message = err instanceof Error ? err.message : "Janus polling failed";
          setTransportError(`Janus stream error: ${message}`);
          const wsReady = await activateWsJpegFallback();
          if (!wsReady) {
            activateMjpegFallback();
          }
        });

        return true;
      } catch {
        if (sessionId && !signal.aborted) {
          void janusPost(`/${sessionId}`, { janus: "destroy" }, signal).catch(() => undefined);
        }
        closeJanusTransport();
        return false;
      }
    };

    const bootstrapVideo = async () => {
      if (disableWebRtc) {
        setTransportError("WebRTC disabled by VITE_DISABLE_WEBRTC; using fallback transport");
        const wsReady = await activateWsJpegFallback();
        if (!wsReady) {
          activateMjpegFallback();
        }
        return;
      }

      setTransportMode("connecting");
      const janusReady = await startJanusTransport();
      if (cancelled) {
        return;
      }
      if (!janusReady) {
        setTransportError("Janus WebRTC unavailable; using fallback transport");
        const wsReady = await activateWsJpegFallback();
        if (!wsReady) {
          activateMjpegFallback();
        }
      }
    };

    bootstrapVideo();

    return () => {
      cancelled = true;
      closeJanusTransport();
      closeWsVideo();
    };
  }, [active, disableWebRtc, resolvedFallbackSrc, resolvedJanusHttpUrl, resolvedJanusMountpoint, resolvedVideoWsUrl]);

  useEffect(() => {
    if (!active) {
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
      const img = mjpegImgRef.current;
      const container = containerRef.current;
      if (!canvas || !container) {
        rafRef.current = window.requestAnimationFrame(draw);
        return;
      }

      const isImageFallback =
        transportModeRef.current === "mjpeg" || transportModeRef.current === "wsjpeg";

      // Determine source dimensions from the active media element
      let nativeW = 0;
      let nativeH = 0;
      if (isImageFallback && img) {
        nativeW = img.naturalWidth;
        nativeH = img.naturalHeight;
      } else if (video) {
        nativeW = video.videoWidth;
        nativeH = video.videoHeight;
      }

      const rect = container.getBoundingClientRect();
      const cssWidth = Math.max(1, Math.round(rect.width));
      const cssHeight = Math.max(1, Math.round(rect.height));
      const sourceWidth = Math.max(1, Math.round(nativeW || cssWidth));
      const sourceHeight = Math.max(1, Math.round(nativeH || cssHeight));
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

      const latest = latestMetadataRef.current;
      const sizeKey = `${cssWidth}x${cssHeight}:${sourceWidth}x${sourceHeight}:${resizeKeyRef.current}:${pixelPerfect ? "pp" : "fit"}`;
      const latestFrameId = latest?.frame_id ?? -1;
      const cache = drawCacheRef.current;
      const shouldDraw =
        cache.frameId !== latestFrameId ||
        cache.sizeKey !== sizeKey ||
        cache.focusedTrackId !== focusedTrackId;
      if (!shouldDraw || !active) {
        rafRef.current = window.requestAnimationFrame(draw);
        return;
      }

      cache.frameId = latestFrameId;
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
        const [rawX, rawY, raw3, raw4] = track.bbox;
        const xywhFits =
          raw3 > 0 &&
          raw4 > 0 &&
          rawX >= 0 &&
          rawY >= 0 &&
          rawX + raw3 <= sourceWidth + 2 &&
          rawY + raw4 <= sourceHeight + 2;
        const xyxyFits =
          raw3 > rawX &&
          raw4 > rawY &&
          rawX >= 0 &&
          rawY >= 0 &&
          raw3 <= sourceWidth + 2 &&
          raw4 <= sourceHeight + 2;
        const useXyxy = xyxyFits && (!xywhFits || !API.USE_MOCK);
        const bx = rawX;
        const by = rawY;
        const bw = useXyxy ? raw3 - rawX : raw3;
        const bh = useXyxy ? raw4 - rawY : raw4;
        const x = offsetX + bx * scaleX;
        const y = offsetY + by * scaleY;
        const w = bw * scaleX;
        const h = bh * scaleY;
        if (w <= 1 || h <= 1) {
          continue;
        }

        const isFocused = focusedTrackId !== null && focusedTrackId === track.track_id;
        const accent = isFocused ? "rgba(255, 198, 89, 1)" : "rgba(124, 92, 255, 0.95)";
        const fill = isFocused ? "rgba(255, 198, 89, 0.13)" : "rgba(124, 92, 255, 0.11)";

        ctx.save();
        ctx.shadowBlur = 14;
        ctx.shadowColor = "rgba(0, 0, 0, 0.35)";
        ctx.fillStyle = fill;
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.3;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
        ctx.restore();

        if (track.thumb) {
          const image = getOrCreateImage(track.thumb, thumbCacheRef.current);
          if (image.complete) {
            ctx.save();
            ctx.shadowBlur = 8;
            ctx.shadowColor = "rgba(0, 0, 0, 0.28)";
            ctx.drawImage(image, x + 6, y + 6, THUMB_SIZE, THUMB_SIZE);
            ctx.strokeStyle = "rgba(255,255,255,0.7)";
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 6, y + 6, THUMB_SIZE, THUMB_SIZE);
            ctx.restore();
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
        style={{ display: transportMode === "janus" ? "block" : "none" }}
        aria-label="Live camera stream"
      />
      <img
        ref={mjpegImgRef}
        alt="Fallback live stream"
        className="h-full w-full object-contain"
        style={{ display: transportMode === "mjpeg" || transportMode === "wsjpeg" ? "block" : "none" }}
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
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
          Live stream fallback active ({transportError})
        </div>
      ) : null}
      {realtime.error ? (
        <div className="pointer-events-none absolute right-3 top-14 rounded-lg bg-rose-400/90 px-3 py-2 text-xs font-medium text-slate-950 shadow-soft">
          Metadata channel reconnecting...
        </div>
      ) : null}
    </div>
  );
}
