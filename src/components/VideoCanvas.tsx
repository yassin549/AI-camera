import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  HttpError,
  postWebRtcOffer,
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
  webrtcOfferUrl?: string;
  fallbackSrc?: string;
  onTrackFocus?: (track: TrackPayload | null) => void;
  onRosterChange?: (tracks: TrackPayload[]) => void;
}

type TransportMode = "connecting" | "webrtc" | "mjpeg";

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

function waitForIceGatheringComplete(pc: RTCPeerConnection, timeoutMs = 1500): Promise<void> {
  if (pc.iceGatheringState === "complete") {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const timer = window.setTimeout(() => {
      pc.removeEventListener("icegatheringstatechange", onStateChange);
      resolve();
    }, timeoutMs);
    const onStateChange = () => {
      if (pc.iceGatheringState === "complete") {
        window.clearTimeout(timer);
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
  webrtcOfferUrl = API.WEBRTC_OFFER,
  fallbackSrc = API.MJPEG_FALLBACK,
  onTrackFocus,
  onRosterChange
}: VideoCanvasProps): JSX.Element {
  const resolvedWsUrl = useMemo(() => metadataUrl ?? resolveMetadataWsUrl(), [metadataUrl]);
  const resolvedWebRtcOfferUrl = useMemo(() => resolveApiUrl(webrtcOfferUrl), [webrtcOfferUrl]);
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
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
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

  const closePeerConnection = () => {
    const pc = peerConnectionRef.current;
    if (!pc) {
      return;
    }
    pc.ontrack = null;
    pc.oniceconnectionstatechange = null;
    pc.close();
    peerConnectionRef.current = null;
  };

  const activateFallback = () => {
    closePeerConnection();
    const video = videoRef.current;
    if (video) {
      video.srcObject = null;
      video.removeAttribute("src");
      video.style.display = "none";
    }
    const img = mjpegImgRef.current;
    if (img) {
      img.src = resolvedFallbackSrc;
      img.style.display = "block";
    }
    transportModeRef.current = "mjpeg";
    setTransportMode("mjpeg");
  };

  useEffect(() => {
    let cancelled = false;
    const video = videoRef.current;
    const img = mjpegImgRef.current;
    if (!active) {
      closePeerConnection();
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

    const bootstrapWebRtc = async () => {
      if (disableWebRtc) {
        activateFallback();
        return;
      }
      if (typeof RTCPeerConnection === "undefined") {
        setTransportError("WebRTC is unavailable in this browser");
        activateFallback();
        return;
      }
      setTransportMode("connecting");
      try {
        const pc = new RTCPeerConnection();
        peerConnectionRef.current = pc;
        pc.addTransceiver("video", { direction: "recvonly" });
        pc.ontrack = (event) => {
          if (event.streams[0]) {
            video.style.display = "block";
            if (img) {
              img.removeAttribute("src");
              img.style.display = "none";
            }
            video.srcObject = event.streams[0];
            video.play().catch(() => {
              // Browser autoplay policy can block without user interaction.
            });
          }
        };
        pc.oniceconnectionstatechange = () => {
          if (pc.iceConnectionState === "failed" || pc.iceConnectionState === "disconnected") {
            activateFallback();
          }
        };

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGatheringComplete(pc);

        const answer = await postWebRtcOffer(
          {
            sdp: pc.localDescription?.sdp ?? offer.sdp ?? "",
            type: "offer"
          },
          resolvedWebRtcOfferUrl
        );
        if (cancelled) {
          return;
        }

        await pc.setRemoteDescription(answer);
        setTransportError(null);
        transportModeRef.current = "webrtc";
        setTransportMode("webrtc");
      } catch (err) {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : "WebRTC signaling failed";
        const expectedFallback =
          (err instanceof HttpError && err.status === 503) ||
          /disabled \(aiortc unavailable\)/i.test(message);
        setTransportError(expectedFallback ? null : message);
        activateFallback();
      }
    };

    bootstrapWebRtc();

    return () => {
      cancelled = true;
      closePeerConnection();
    };
  }, [active, disableWebRtc, resolvedFallbackSrc, resolvedWebRtcOfferUrl]);

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
    let mjpegFrameKey = 0;
    const draw = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const img = mjpegImgRef.current;
      const container = containerRef.current;
      if (!canvas || !container) {
        rafRef.current = window.requestAnimationFrame(draw);
        return;
      }

      const isMjpeg = transportModeRef.current === "mjpeg";

      // Determine source dimensions from the active media element
      let nativeW = 0;
      let nativeH = 0;
      if (isMjpeg && img) {
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

      // For MJPEG mode, always redraw because the <img> updates in-place
      const latest = latestMetadataRef.current;
      const currentMjpegKey = isMjpeg ? ++mjpegFrameKey : 0;
      const sizeKey = `${cssWidth}x${cssHeight}:${sourceWidth}x${sourceHeight}:${resizeKeyRef.current
        }:${pixelPerfect ? "pp" : "fit"}:${isMjpeg ? currentMjpegKey : ""}`;
      const latestFrameId = latest?.frame_id ?? -1;
      const cache = drawCacheRef.current;
      const shouldDraw =
        isMjpeg ||
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

      // In MJPEG mode, draw the <img> onto the canvas as the video background
      if (isMjpeg && img && img.complete && img.naturalWidth > 0) {
        const fitScale = Math.min(cssWidth / img.naturalWidth, cssHeight / img.naturalHeight);
        const drawW = img.naturalWidth * fitScale;
        const drawH = img.naturalHeight * fitScale;
        const drawX = (cssWidth - drawW) / 2;
        const drawY = (cssHeight - drawH) / 2;
        ctx.drawImage(img, drawX, drawY, drawW, drawH);
      }

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
        style={{ display: transportMode === "mjpeg" ? "none" : "block" }}
        aria-label="Live camera stream"
      />
      {/* Hidden MJPEG <img> — browsers can render multipart/x-mixed-replace on <img> but not <video> */}
      <img
        ref={mjpegImgRef}
        alt="MJPEG live stream"
        style={{ display: "none", position: "absolute", width: 0, height: 0, pointerEvents: "none" }}
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
          WebRTC failed, MJPEG fallback active ({transportError})
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
