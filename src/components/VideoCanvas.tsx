import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  resolveMetadataLatestUrl,
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
  metadataLatestUrl?: string;
  janusHttpUrl?: string;
  janusMountpoint?: number;
  disableBackendVideo?: boolean;
  disableJanus?: boolean;
  onTrackFocus?: (track: TrackPayload | null) => void;
  onRosterChange?: (tracks: TrackPayload[]) => void;
}

type TransportMode = "connecting" | "janus" | "wsjpeg" | "error";
type TransportCandidate = "janus" | "wsjpeg";
type JanusStep = "create" | "attach" | "watch" | "offer" | "answer" | "first-track";

class JanusStepError extends Error {
  step: JanusStep;
  code: string;

  constructor(step: JanusStep, code: string, message: string) {
    super(message);
    this.name = "JanusStepError";
    this.step = step;
    this.code = code;
  }
}

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
  if (/^(https?|wss?):\/\//i.test(pathOrUrl) || pathOrUrl.startsWith("blob:") || pathOrUrl.startsWith("data:")) {
    return pathOrUrl;
  }
  if (!API.REST_BASE) {
    return pathOrUrl;
  }
  const base = API.REST_BASE.replace(/\/+$/, "");
  const path = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${base}${path}`;
}

function toWebSocketUrl(url: string): string {
  if (!url) {
    return url;
  }
  try {
    const parsed = new URL(url, window.location.href);
    if (parsed.protocol === "https:") {
      parsed.protocol = "wss:";
    } else if (parsed.protocol === "http:") {
      parsed.protocol = "ws:";
    }
    return parsed.toString();
  } catch {
    return url;
  }
}

function isNgrokFreeUrl(input: string): boolean {
  try {
    const host = new URL(input, window.location.href).hostname.toLowerCase();
    return host.endsWith(".ngrok-free.app") || host.endsWith(".ngrok-free.dev");
  } catch {
    return false;
  }
}

function appendNgrokSkipParam(input: string): string {
  if (!input || !isNgrokFreeUrl(input)) {
    return input;
  }
  try {
    const parsed = new URL(input, window.location.href);
    if (!parsed.searchParams.has("ngrok-skip-browser-warning")) {
      parsed.searchParams.set("ngrok-skip-browser-warning", "1");
    }
    return parsed.toString();
  } catch {
    const [base, hash = ""] = input.split("#", 2);
    if (base.includes("ngrok-skip-browser-warning=")) {
      return input;
    }
    const separator = base.includes("?") ? "&" : "?";
    const next = `${base}${separator}ngrok-skip-browser-warning=1`;
    return hash ? `${next}#${hash}` : next;
  }
}

function nextTx(): string {
  return Math.random().toString(36).slice(2, 12);
}

function computeTracksKey(payload: MetadataPayload | null): string {
  if (!payload || payload.tracks.length === 0) {
    return "none";
  }
  const trackKey = payload.tracks
    .map((track) => {
      const [x1, y1, x2, y2] = track.bbox;
      return `${track.track_id}:${x1},${y1},${x2},${y2}:${track.identity_id ?? "n"}:${track.label}:${track.muted ? 1 : 0}:${track.age_frames ?? 0}`;
    })
    .join("|");
  return `${payload.frame_id}:${trackKey}`;
}

function computeRosterKey(tracks: TrackPayload[]): string {
  if (!tracks.length) {
    return "none";
  }
  return tracks
    .map((track) => `${track.track_id}:${track.identity_id ?? "n"}:${track.label}:${track.muted ? 1 : 0}`)
    .join("|");
}

function percentile(values: number[], p: number): number | null {
  if (!values.length) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const clamped = Math.max(0, Math.min(100, p));
  const rank = (clamped / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.min(sorted.length - 1, lo + 1);
  const frac = rank - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

function transportLabel(transport: TransportCandidate): string {
  return transport === "wsjpeg" ? "WS-JPEG" : "JANUS";
}

export function VideoCanvas({
  className,
  active = true,
  pixelPerfect = true,
  focusedTrackId = null,
  metadataUrl,
  metadataLatestUrl,
  janusHttpUrl = API.JANUS_HTTP,
  janusMountpoint = API.JANUS_MOUNTPOINT,
  disableBackendVideo = API.DISABLE_BACKEND_VIDEO,
  disableJanus = API.DISABLE_JANUS,
  onTrackFocus,
  onRosterChange
}: VideoCanvasProps): JSX.Element {
  const resolvedWsUrl = useMemo(
    () => withApiKeyQuery(metadataUrl ?? resolveMetadataWsUrl()),
    [metadataUrl]
  );
  const resolvedLatestMetadataUrl = useMemo(
    () => withApiKeyQuery(metadataLatestUrl ?? resolveMetadataLatestUrl()),
    [metadataLatestUrl]
  );
  const resolvedJanusHttpUrl = useMemo(() => resolveApiUrl(janusHttpUrl).replace(/\/+$/, ""), [janusHttpUrl]);
  const resolvedJanusWsUrl = useMemo(() => {
    if (!API.JANUS_WS) {
      return "";
    }
    return appendNgrokSkipParam(toWebSocketUrl(resolveApiUrl(API.JANUS_WS)));
  }, []);
  const resolvedWsJpegUrl = useMemo(() => {
    const base = toWebSocketUrl(resolveApiUrl("/api/media/ws"));
    const remoteFps = isNgrokFreeUrl(API.REST_BASE) ? API.WSJPEG_FPS_REMOTE : API.WSJPEG_FPS_LOCAL;
    const adaptive = API.WSJPEG_ADAPTIVE ? "1" : "0";
    const separator = base.includes("?") ? "&" : "?";
    return withApiKeyQuery(`${base}${separator}fps=${remoteFps}&adaptive=${adaptive}`);
  }, []);
  const resolvedJanusMountpoint = useMemo(
    () => (Number.isFinite(janusMountpoint) && Number(janusMountpoint) > 0 ? Number(janusMountpoint) : 1),
    [janusMountpoint]
  );
  const janusFirstTrackTimeoutMs = useMemo(
    () =>
      Math.max(
        1000,
        Number.isFinite(API.JANUS_FIRST_TRACK_TIMEOUT_MS) ? Number(API.JANUS_FIRST_TRACK_TIMEOUT_MS) : 15000
      ),
    []
  );
  const wsJpegRenderMinIntervalMs = useMemo(() => {
    const maxFps = Number.isFinite(API.WSJPEG_RENDER_MAX_FPS) ? Number(API.WSJPEG_RENDER_MAX_FPS) : 0;
    if (maxFps <= 0) {
      return 0;
    }
    return Math.max(0, 1000 / maxFps);
  }, []);
  const configuredTransportPlan = API.VIDEO_TRANSPORT_PLAN as readonly TransportCandidate[];
  const authHeaders = useMemo(() => withApiKeyHeaders(), []);
  const effectiveTransportPlan = useMemo(() => {
    if (!configuredTransportPlan.length) {
      return [] as TransportCandidate[];
    }
    if (!disableBackendVideo) {
      return configuredTransportPlan as TransportCandidate[];
    }
    return configuredTransportPlan.filter((candidate) => candidate !== "wsjpeg") as TransportCandidate[];
  }, [configuredTransportPlan, disableBackendVideo]);
  const configuredTransportPlanLabel = useMemo(
    () => (effectiveTransportPlan.length ? effectiveTransportPlan.map((item) => transportLabel(item)).join(" -> ") : "NONE"),
    [effectiveTransportPlan]
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
  const rosterPendingRef = useRef<TrackPayload[] | null>(null);
  const rosterFlushTimerRef = useRef<number | null>(null);
  const metricsRef = useRef<CanvasMetrics>({ cssWidth: 0, cssHeight: 0, sourceWidth: 0, sourceHeight: 0 });
  const containerSizeRef = useRef<{ width: number; height: number }>({ width: 1, height: 1 });
  const drawRafRef = useRef<number | null>(null);
  const drawScheduledRef = useRef(false);
  const requestDrawRef = useRef<() => void>(() => undefined);
  const metadataLagSamplesRef = useRef<number[]>([]);
  const overlayIntervalsRef = useRef<number[]>([]);
  const lastOverlayDrawAtRef = useRef<number | null>(null);
  const janusConnectStartedAtRef = useRef<number | null>(null);
  const janusTtffMsRef = useRef<number | null>(null);
  const janusDiagnosticsRef = useRef<string[]>([]);
  const perfPostInflightRef = useRef(false);
  const janusPcRef = useRef<RTCPeerConnection | null>(null);
  const janusWsRef = useRef<WebSocket | null>(null);
  const janusTransportRef = useRef<"http" | "ws" | null>(null);
  const wsJpegSocketRef = useRef<WebSocket | null>(null);
  const wsJpegObjectUrlRef = useRef<string | null>(null);
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
    latestUrl: resolvedLatestMetadataUrl,
    pollIntervalMs: API.METADATA_POLL_MS,
    headers: authHeaders,
    enabled: active,
    onMessage: (payload) => {
      const resolvedTracks = payload.tracks.map((track) => ({
        ...track,
        thumb: track.thumb ? withApiKeyQuery(resolveApiUrl(track.thumb)) : undefined
      }));
      const resolvedPayload: MetadataPayload = {
        ...payload,
        tracks: resolvedTracks
      };
      latestMetadataRef.current = resolvedPayload;
      const captureTsUnix = Number(resolvedPayload.capture_ts_unix ?? NaN);
      const payloadLagMs = Number(resolvedPayload.metadata_lag_ms ?? NaN);
      let estimatedLagMs: number | null = null;
      if (Number.isFinite(captureTsUnix) && captureTsUnix > 0) {
        estimatedLagMs = Date.now() - captureTsUnix * 1000;
      } else if (Number.isFinite(payloadLagMs) && payloadLagMs >= 0) {
        estimatedLagMs = payloadLagMs;
      }
      if (estimatedLagMs !== null && Number.isFinite(estimatedLagMs) && estimatedLagMs >= 0 && estimatedLagMs < 120000) {
        const next = metadataLagSamplesRef.current;
        next.push(estimatedLagMs);
        if (next.length > 256) {
          next.splice(0, next.length - 256);
        }
      }
      requestDrawRef.current();
      if (onRosterChange) {
        rosterPendingRef.current = resolvedTracks;
        if (rosterFlushTimerRef.current === null) {
          rosterFlushTimerRef.current = window.setTimeout(() => {
            rosterFlushTimerRef.current = null;
            const tracks = rosterPendingRef.current ?? [];
            rosterPendingRef.current = null;
            const nextKey = computeRosterKey(tracks);
            const now = performance.now();
            const prev = rosterEmitRef.current;
            if (nextKey !== prev.key || now - prev.at >= 300) {
              rosterEmitRef.current = { key: nextKey, at: now };
              onRosterChange(tracks);
            }
          }, 120);
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
    janusConnectStartedAtRef.current = null;
    const janusPc = janusPcRef.current;
    if (!janusPc) {
      // continue: session may still need cleanup.
    } else {
      janusPc.ontrack = null;
      janusPc.onicecandidate = null;
      janusPc.close();
      janusPcRef.current = null;
    }
    const janusWs = janusWsRef.current;
    if (janusWs) {
      try {
        janusWs.onopen = null;
        janusWs.onmessage = null;
        janusWs.onerror = null;
        janusWs.onclose = null;
      } catch {
        // Ignore WS handler cleanup errors.
      }
    }
    const wsJpegSocket = wsJpegSocketRef.current;
    if (wsJpegSocket) {
      try {
        wsJpegSocket.onopen = null;
        wsJpegSocket.onmessage = null;
        wsJpegSocket.onerror = null;
        wsJpegSocket.onclose = null;
        wsJpegSocket.close();
      } catch {
        // Ignore websocket close errors during transport teardown.
      }
      wsJpegSocketRef.current = null;
    }
    const objectUrl = wsJpegObjectUrlRef.current;
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      wsJpegObjectUrlRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      video.onloadeddata = null;
      video.onerror = null;
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
    const transport = janusTransportRef.current;
    janusSessionRef.current = null;
    janusHandleRef.current = null;
    janusTransportRef.current = null;
    if (!sessionId) {
      if (janusWs) {
        try {
          janusWs.close();
        } catch {
          // Ignore WS close errors.
        }
        janusWsRef.current = null;
      }
      return;
    }

    if (transport === "ws" && janusWs && janusWs.readyState === WebSocket.OPEN) {
      try {
        janusWs.send(JSON.stringify({ janus: "destroy", session_id: sessionId, transaction: nextTx() }));
      } catch {
        // Ignore destroy failures; session will expire server-side.
      }
    } else if (resolvedJanusHttpUrl) {
      const payload = {
        janus: "destroy",
        transaction: nextTx(),
      };
      const destroyHeaders = withApiKeyHeaders();
      destroyHeaders.set("Content-Type", "application/json");
      void fetch(`${resolvedJanusHttpUrl}/${sessionId}`, {
        method: "POST",
        headers: destroyHeaders,
        body: JSON.stringify(payload),
      }).catch(() => undefined);
    }
    if (janusWs) {
      try {
        janusWs.close();
      } catch {
        // Ignore WS close errors.
      }
      janusWsRef.current = null;
    }
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
      mjpeg.onload = null;
      mjpeg.onerror = null;
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

    const normalizeCodeToken = (input: unknown): string => {
      const token = String(input ?? "unknown")
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
      return token || "unknown";
    };

    const normalizeSnippet = (raw: string): string => {
      if (!raw) {
        return "";
      }
      return raw.replace(/\s+/g, " ").trim().slice(0, 180);
    };

    const classifyHttpStatus = (status: number): string => {
      if (status === 401 || status === 403) {
        return "proxy_auth_failed";
      }
      if (status === 404) {
        return "proxy_path_not_found";
      }
      if (status === 502 || status === 503 || status === 504) {
        return "proxy_upstream_unavailable";
      }
      return `http_${status}`;
    };

    const pushJanusDiagnostic = (step: JanusStep, code: string, detail: string, ok: boolean) => {
      const entry = `[janus:${step}] ${code}${detail ? ` | ${detail}` : ""}`;
      const history = janusDiagnosticsRef.current;
      history.push(entry);
      if (history.length > 40) {
        history.splice(0, history.length - 40);
      }
      if (ok) {
        console.info(entry);
      } else {
        console.error(entry);
      }
    };

    const parseJanusPayload = async (response: Response, step: JanusStep): Promise<Record<string, unknown>> => {
      const raw = await response.text();
      const trimmed = raw.trim();
      if (!trimmed) {
        throw new JanusStepError(step, `${step}_empty_response`, "Janus returned an empty response body");
      }
      let payloadAny: unknown;
      try {
        payloadAny = JSON.parse(trimmed) as unknown;
      } catch {
        throw new JanusStepError(
          step,
          `${step}_non_json_response`,
          `Expected Janus JSON but received: ${normalizeSnippet(trimmed)}`
        );
      }
      if (!payloadAny || typeof payloadAny !== "object" || Array.isArray(payloadAny)) {
        throw new JanusStepError(step, `${step}_invalid_payload`, "Janus payload has invalid shape");
      }
      return payloadAny as Record<string, unknown>;
    };

    const janusPost = async (
      path: string,
      body: Record<string, unknown>,
      signal: AbortSignal,
      step: JanusStep
    ): Promise<Record<string, unknown>> => {
      const headers = new Headers();
      headers.set("Content-Type", "application/json");
      let response: Response;
      try {
        response = await fetch(withApiKeyQuery(`${resolvedJanusHttpUrl}${path}`), {
          method: "POST",
          headers,
          body: JSON.stringify({ ...body, transaction: nextTx() }),
          signal,
        });
      } catch (err: unknown) {
        const reason = err instanceof Error ? err.message : "request failed";
        throw new JanusStepError(step, `${step}_network_unreachable`, reason);
      }
      if (!response.ok) {
        const raw = await response.text().catch(() => "");
        const snippet = normalizeSnippet(raw);
        const detail = snippet ? `HTTP ${response.status} ${snippet}` : `HTTP ${response.status}`;
        throw new JanusStepError(step, `${step}_${classifyHttpStatus(response.status)}`, detail);
      }
      const payload = await parseJanusPayload(response, step);
      if (payload.janus === "error") {
        const error = payload.error as { code?: number; reason?: string } | undefined;
        const apiCode = normalizeCodeToken(error?.code ?? "unknown");
        const reason = normalizeSnippet(error?.reason ?? "");
        throw new JanusStepError(
          step,
          `${step}_janus_error_${apiCode}`,
          reason || `Janus reported error code ${error?.code ?? "unknown"}`
        );
      }
      return payload;
    };

    const startJanusTransportHttp = async (): Promise<{ ok: boolean; error: string }> => {
      janusConnectStartedAtRef.current = performance.now();
      janusTtffMsRef.current = null;
      janusDiagnosticsRef.current = [];
      janusTransportRef.current = "http";
      if (!resolvedJanusHttpUrl || typeof RTCPeerConnection === "undefined") {
        return { ok: false, error: "janus_unavailable: Janus endpoint or browser WebRTC unavailable" };
      }
      const abort = new AbortController();
      const signal = abort.signal;
      janusAbortRef.current = abort;
      let sessionId: number | null = null;
      let handleId: number | null = null;
      let sawOffer = false;
      let sentAnswer = false;
      let resolvedFirstTrack = false;
      let firstTrackResolve: ((value: boolean) => void) | null = null;
      let firstTrackTimeout: number | null = null;
      let firstTrackFailure: JanusStepError | null = null;
      const settleFirstTrack = (value: boolean) => {
        if (resolvedFirstTrack) {
          return;
        }
        resolvedFirstTrack = true;
        if (firstTrackTimeout !== null) {
          window.clearTimeout(firstTrackTimeout);
          firstTrackTimeout = null;
        }
        if (firstTrackResolve) {
          firstTrackResolve(value);
          firstTrackResolve = null;
        }
      };

      try {
        const created = await janusPost("", { janus: "create" }, signal, "create");
        sessionId = Number((created.data as { id?: number } | undefined)?.id ?? -1);
        if (!Number.isFinite(sessionId) || sessionId <= 0) {
          throw new JanusStepError("create", "create_invalid_session_id", "Janus create response missing valid session id");
        }
        pushJanusDiagnostic("create", "ok", `session_id=${sessionId}`, true);

        const attached = await janusPost(
          `/${sessionId}`,
          { janus: "attach", plugin: "janus.plugin.streaming" },
          signal,
          "attach"
        );
        handleId = Number((attached.data as { id?: number } | undefined)?.id ?? -1);
        if (!Number.isFinite(handleId) || handleId <= 0) {
          throw new JanusStepError("attach", "attach_invalid_handle_id", "Janus attach response missing valid handle id");
        }
        pushJanusDiagnostic("attach", "ok", `handle_id=${handleId}`, true);

        janusSessionRef.current = sessionId;
        janusHandleRef.current = handleId;
        const pc = new RTCPeerConnection();
        janusPcRef.current = pc;

        const firstTrack = new Promise<boolean>((resolve) => {
          firstTrackResolve = resolve;
          firstTrackTimeout = window.setTimeout(() => {
            if (!sawOffer) {
              firstTrackFailure = new JanusStepError(
                "offer",
                "offer_not_received",
                `No Janus SDP offer received within ${Math.round(janusFirstTrackTimeoutMs)}ms`
              );
            } else if (sentAnswer) {
              firstTrackFailure = new JanusStepError(
                "first-track",
                "signal_ok_media_failed",
                `Signaling completed but no media track received within ${Math.round(janusFirstTrackTimeoutMs)}ms`
              );
            } else {
              firstTrackFailure = new JanusStepError(
                "answer",
                "answer_not_completed",
                `Offer received but local answer was not completed within ${Math.round(janusFirstTrackTimeoutMs)}ms`
              );
            }
            pushJanusDiagnostic(
              firstTrackFailure.step,
              firstTrackFailure.code,
              firstTrackFailure.message,
              false
            );
            settleFirstTrack(false);
          }, janusFirstTrackTimeoutMs);

          pc.ontrack = (event) => {
            if (!event.streams[0]) {
              return;
            }
            settleFirstTrack(true);
            showVideo();
            video.srcObject = event.streams[0];
            video.play().catch(() => undefined);
            if (janusTtffMsRef.current === null && janusConnectStartedAtRef.current !== null) {
              janusTtffMsRef.current = Math.max(0, performance.now() - janusConnectStartedAtRef.current);
            }
            pushJanusDiagnostic("first-track", "ok", "Remote media track received", true);
            transportModeRef.current = "janus";
            setTransportMode("janus");
            setTransportError(null);
          };
        });

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
            signal,
            "answer"
          ).catch(() => undefined);
        };

        await janusPost(
          `/${sessionId}/${handleId}`,
          { janus: "message", body: { request: "watch", id: resolvedJanusMountpoint } },
          signal,
          "watch"
        );
        pushJanusDiagnostic("watch", "ok", `mountpoint=${resolvedJanusMountpoint}`, true);

        janusKeepaliveRef.current = window.setInterval(() => {
          if (signal.aborted || !janusSessionRef.current) {
            return;
          }
          void janusPost(
            `/${janusSessionRef.current}`,
            { janus: "keepalive" },
            signal,
            "watch"
          ).catch(() => undefined);
        }, 25000);

        const pollEvents = async () => {
          while (!signal.aborted && janusSessionRef.current && janusHandleRef.current) {
            const rid = Date.now();
            let response: Response;
            try {
              response = await fetch(
                withApiKeyQuery(`${resolvedJanusHttpUrl}/${janusSessionRef.current}?rid=${rid}&maxev=10`),
                { signal }
              );
            } catch (err: unknown) {
              const reason = err instanceof Error ? err.message : "poll request failed";
              throw new JanusStepError("offer", "offer_poll_network_unreachable", reason);
            }
            if (!response.ok) {
              const raw = await response.text().catch(() => "");
              const snippet = normalizeSnippet(raw);
              const detail = snippet ? `HTTP ${response.status} ${snippet}` : `HTTP ${response.status}`;
              throw new JanusStepError("offer", `offer_${classifyHttpStatus(response.status)}`, detail);
            }
            const event = await parseJanusPayload(response, "offer");
            const eventSender = Number(event.sender ?? -1);

            if (
              event.janus === "event" &&
              eventSender === janusHandleRef.current &&
              event.jsep &&
              typeof event.jsep === "object" &&
              janusPcRef.current
            ) {
              if (!sawOffer) {
                sawOffer = true;
                pushJanusDiagnostic("offer", "ok", "Received remote SDP offer", true);
              }
              const remote = event.jsep as RTCSessionDescriptionInit;
              try {
                await janusPcRef.current.setRemoteDescription(remote);
              } catch (err: unknown) {
                const reason = err instanceof Error ? err.message : "setRemoteDescription failed";
                throw new JanusStepError("offer", "offer_set_remote_failed", reason);
              }
              try {
                const answer = await janusPcRef.current.createAnswer();
                await janusPcRef.current.setLocalDescription(answer);
              } catch (err: unknown) {
                const reason = err instanceof Error ? err.message : "create/set local answer failed";
                throw new JanusStepError("answer", "answer_local_description_failed", reason);
              }
              await janusPost(
                `/${janusSessionRef.current}/${janusHandleRef.current}`,
                {
                  janus: "message",
                  body: { request: "start" },
                  jsep: janusPcRef.current.localDescription,
                },
                signal,
                "answer"
              );
              if (!sentAnswer) {
                sentAnswer = true;
                pushJanusDiagnostic("answer", "ok", "Submitted local SDP answer to Janus", true);
              }
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
          if (err instanceof JanusStepError) {
            firstTrackFailure = err;
            pushJanusDiagnostic(err.step, err.code, err.message, false);
          } else {
            const message = err instanceof Error ? err.message : "Janus polling failed";
            firstTrackFailure = new JanusStepError("offer", "offer_poll_failed", message);
            pushJanusDiagnostic("offer", "offer_poll_failed", message, false);
          }
          settleFirstTrack(false);
        });

        const gotTrack = await firstTrack;
        if (!gotTrack) {
          if (firstTrackFailure) {
            throw firstTrackFailure;
          }
          throw new JanusStepError(
            "first-track",
            "first_track_timeout",
            `No media track received within ${Math.round(janusFirstTrackTimeoutMs)}ms`
          );
        }
        return { ok: true, error: "" };
      } catch (err: unknown) {
        const normalizedError =
          err instanceof JanusStepError
            ? err
            : new JanusStepError(
                "first-track",
                "janus_negotiation_failed",
                err instanceof Error ? err.message : "Janus negotiation failed"
              );
        if (!(err instanceof JanusStepError)) {
          pushJanusDiagnostic(normalizedError.step, normalizedError.code, normalizedError.message, false);
        }
        if (sessionId && !signal.aborted) {
          void janusPost(`/${sessionId}`, { janus: "destroy" }, signal, "create").catch(() => undefined);
        }
        closeTransport();
        return {
          ok: false,
          error: `${normalizedError.code} [${normalizedError.step}] ${normalizedError.message}`,
        };
      }
    };

    const startJanusTransportWs = async (): Promise<{ ok: boolean; error: string }> => {
      janusConnectStartedAtRef.current = performance.now();
      janusTtffMsRef.current = null;
      janusDiagnosticsRef.current = [];
      if (!resolvedJanusWsUrl || typeof RTCPeerConnection === "undefined") {
        return { ok: false, error: "janus_ws_unavailable: Janus WS endpoint or browser WebRTC unavailable" };
      }

      const abort = new AbortController();
      const signal = abort.signal;
      janusAbortRef.current = abort;
      let sessionId: number | null = null;
      let handleId: number | null = null;
      let sawOffer = false;
      let sentAnswer = false;
      let resolvedFirstTrack = false;
      let firstTrackResolve: ((value: boolean) => void) | null = null;
      let firstTrackTimeout: number | null = null;
      let firstTrackFailure: JanusStepError | null = null;
      const settleFirstTrack = (value: boolean) => {
        if (resolvedFirstTrack) {
          return;
        }
        resolvedFirstTrack = true;
        if (firstTrackTimeout !== null) {
          window.clearTimeout(firstTrackTimeout);
          firstTrackTimeout = null;
        }
        if (firstTrackResolve) {
          firstTrackResolve(value);
          firstTrackResolve = null;
        }
      };

      const pending = new Map<
        string,
        {
          resolve: (payload: Record<string, unknown>) => void;
          reject: (error: JanusStepError) => void;
          timer: number;
          step: JanusStep;
        }
      >();

      const closePending = (error: JanusStepError) => {
        for (const [_, entry] of pending.entries()) {
          window.clearTimeout(entry.timer);
          entry.reject(error);
        }
        pending.clear();
      };

      const sendWs = (payload: Record<string, unknown>, step: JanusStep, timeoutMs = 8000) => {
        return new Promise<Record<string, unknown>>((resolve, reject) => {
          if (signal.aborted) {
            reject(new JanusStepError(step, `${step}_aborted`, "Janus WS request aborted"));
            return;
          }
          const ws = janusWsRef.current;
          if (!ws || ws.readyState !== WebSocket.OPEN) {
            reject(new JanusStepError(step, `${step}_ws_unavailable`, "Janus WebSocket not connected"));
            return;
          }
          const tx = nextTx();
          const timer = window.setTimeout(() => {
            pending.delete(tx);
            reject(new JanusStepError(step, `${step}_timeout`, "Janus WS response timed out"));
          }, timeoutMs);
          pending.set(tx, { resolve, reject, timer, step });
          try {
            ws.send(JSON.stringify({ ...payload, transaction: tx }));
          } catch (err: unknown) {
            window.clearTimeout(timer);
            pending.delete(tx);
            const reason = err instanceof Error ? err.message : "Janus WS send failed";
            reject(new JanusStepError(step, `${step}_ws_send_failed`, reason));
          }
        });
      };

      try {
        const ws = new WebSocket(resolvedJanusWsUrl);
        janusWsRef.current = ws;
        janusTransportRef.current = "ws";

        const opened = await new Promise<boolean>((resolve) => {
          const timeout = window.setTimeout(() => resolve(false), 6000);
          ws.onopen = () => {
            window.clearTimeout(timeout);
            resolve(true);
          };
          ws.onerror = () => {
            window.clearTimeout(timeout);
            resolve(false);
          };
        });
        if (!opened || signal.aborted) {
          throw new JanusStepError("create", "ws_connect_failed", "Failed to open Janus WebSocket connection");
        }

        ws.onmessage = async (event) => {
          let payloadAny: unknown;
          try {
            payloadAny = JSON.parse(String(event.data));
          } catch {
            return;
          }
          if (!payloadAny || typeof payloadAny !== "object" || Array.isArray(payloadAny)) {
            return;
          }
          const payload = payloadAny as Record<string, unknown>;
          const tx = String(payload.transaction ?? "");
          if (tx && pending.has(tx)) {
            const entry = pending.get(tx);
            if (!entry) {
              return;
            }
            window.clearTimeout(entry.timer);
            pending.delete(tx);
            if (payload.janus === "error") {
              const error = payload.error as { code?: number; reason?: string } | undefined;
              const apiCode = normalizeCodeToken(error?.code ?? "unknown");
              const reason = normalizeSnippet(error?.reason ?? "");
              entry.reject(
                new JanusStepError(
                  entry.step,
                  `${entry.step}_janus_error_${apiCode}`,
                  reason || `Janus reported error code ${error?.code ?? "unknown"}`
                )
              );
              return;
            }
            entry.resolve(payload);
            return;
          }

          const eventSender = Number(payload.sender ?? -1);
          if (
            payload.janus === "event" &&
            eventSender === janusHandleRef.current &&
            payload.jsep &&
            typeof payload.jsep === "object" &&
            janusPcRef.current
          ) {
            if (!sawOffer) {
              sawOffer = true;
              pushJanusDiagnostic("offer", "ok", "Received remote SDP offer", true);
            }
            const remote = payload.jsep as RTCSessionDescriptionInit;
            try {
              await janusPcRef.current.setRemoteDescription(remote);
            } catch (err: unknown) {
              const reason = err instanceof Error ? err.message : "setRemoteDescription failed";
              firstTrackFailure = new JanusStepError("offer", "offer_set_remote_failed", reason);
              pushJanusDiagnostic("offer", "offer_set_remote_failed", reason, false);
              settleFirstTrack(false);
              return;
            }
            try {
              const answer = await janusPcRef.current.createAnswer();
              await janusPcRef.current.setLocalDescription(answer);
            } catch (err: unknown) {
              const reason = err instanceof Error ? err.message : "create/set local answer failed";
              firstTrackFailure = new JanusStepError("answer", "answer_local_description_failed", reason);
              pushJanusDiagnostic("answer", "answer_local_description_failed", reason, false);
              settleFirstTrack(false);
              return;
            }
            try {
              await sendWs(
                {
                  janus: "message",
                  body: { request: "start" },
                  jsep: janusPcRef.current.localDescription,
                  session_id: janusSessionRef.current ?? undefined,
                  handle_id: janusHandleRef.current ?? undefined
                },
                "answer"
              );
              if (!sentAnswer) {
                sentAnswer = true;
                pushJanusDiagnostic("answer", "ok", "Submitted local SDP answer to Janus", true);
              }
            } catch (err: unknown) {
              const normalized = err instanceof JanusStepError
                ? err
                : new JanusStepError(
                    "answer",
                    "answer_ws_send_failed",
                    err instanceof Error ? err.message : "Failed to send Janus answer"
                  );
              firstTrackFailure = normalized;
              pushJanusDiagnostic(normalized.step, normalized.code, normalized.message, false);
              settleFirstTrack(false);
            }
            return;
          }

          if (
            payload.janus === "trickle" &&
            eventSender === janusHandleRef.current &&
            payload.candidate &&
            janusPcRef.current
          ) {
            const candidate = payload.candidate as { completed?: boolean; candidate?: string };
            try {
              await janusPcRef.current.addIceCandidate(candidate.completed ? null : (candidate as RTCIceCandidateInit));
            } catch {
              // Ignore invalid trickle candidates.
            }
          }
        };

        ws.onclose = () => {
          if (signal.aborted) {
            return;
          }
          const err = new JanusStepError("offer", "ws_closed", "Janus WebSocket closed");
          closePending(err);
          if (!resolvedFirstTrack) {
            firstTrackFailure = err;
            pushJanusDiagnostic(err.step, err.code, err.message, false);
            settleFirstTrack(false);
          }
        };

        const created = await sendWs({ janus: "create" }, "create");
        sessionId = Number((created.data as { id?: number } | undefined)?.id ?? -1);
        if (!Number.isFinite(sessionId) || sessionId <= 0) {
          throw new JanusStepError("create", "create_invalid_session_id", "Janus create response missing valid session id");
        }
        pushJanusDiagnostic("create", "ok", `session_id=${sessionId}`, true);

        const attached = await sendWs({ janus: "attach", plugin: "janus.plugin.streaming", session_id: sessionId }, "attach");
        handleId = Number((attached.data as { id?: number } | undefined)?.id ?? -1);
        if (!Number.isFinite(handleId) || handleId <= 0) {
          throw new JanusStepError("attach", "attach_invalid_handle_id", "Janus attach response missing valid handle id");
        }
        pushJanusDiagnostic("attach", "ok", `handle_id=${handleId}`, true);

        janusSessionRef.current = sessionId;
        janusHandleRef.current = handleId;
        const pc = new RTCPeerConnection();
        janusPcRef.current = pc;

        const firstTrack = new Promise<boolean>((resolve) => {
          firstTrackResolve = resolve;
          firstTrackTimeout = window.setTimeout(() => {
            if (!sawOffer) {
              firstTrackFailure = new JanusStepError(
                "offer",
                "offer_not_received",
                `No Janus SDP offer received within ${Math.round(janusFirstTrackTimeoutMs)}ms`
              );
            } else if (sentAnswer) {
              firstTrackFailure = new JanusStepError(
                "first-track",
                "signal_ok_media_failed",
                `Signaling completed but no media track received within ${Math.round(janusFirstTrackTimeoutMs)}ms`
              );
            } else {
              firstTrackFailure = new JanusStepError(
                "answer",
                "answer_not_completed",
                `Offer received but local answer was not completed within ${Math.round(janusFirstTrackTimeoutMs)}ms`
              );
            }
            pushJanusDiagnostic(firstTrackFailure.step, firstTrackFailure.code, firstTrackFailure.message, false);
            settleFirstTrack(false);
          }, janusFirstTrackTimeoutMs);

          pc.ontrack = (event) => {
            if (!event.streams[0]) {
              return;
            }
            settleFirstTrack(true);
            showVideo();
            video.srcObject = event.streams[0];
            video.play().catch(() => undefined);
            if (janusTtffMsRef.current === null && janusConnectStartedAtRef.current !== null) {
              janusTtffMsRef.current = Math.max(0, performance.now() - janusConnectStartedAtRef.current);
            }
            pushJanusDiagnostic("first-track", "ok", "Remote media track received", true);
            transportModeRef.current = "janus";
            setTransportMode("janus");
            setTransportError(null);
          };
        });

        pc.onicecandidate = (event) => {
          const sid = janusSessionRef.current;
          const hid = janusHandleRef.current;
          if (!sid || !hid || signal.aborted) {
            return;
          }
          void sendWs(
            {
              janus: "trickle",
              candidate: event.candidate ?? { completed: true },
              session_id: sid,
              handle_id: hid
            },
            "answer"
          ).catch(() => undefined);
        };

        await sendWs(
          {
            janus: "message",
            body: { request: "watch", id: resolvedJanusMountpoint },
            session_id: sessionId,
            handle_id: handleId
          },
          "watch"
        );
        pushJanusDiagnostic("watch", "ok", `mountpoint=${resolvedJanusMountpoint}`, true);

        janusKeepaliveRef.current = window.setInterval(() => {
          if (signal.aborted || !janusSessionRef.current) {
            return;
          }
          void sendWs({ janus: "keepalive", session_id: janusSessionRef.current }, "watch").catch(() => undefined);
        }, 25000);

        const gotTrack = await firstTrack;
        if (!gotTrack) {
          if (firstTrackFailure) {
            throw firstTrackFailure;
          }
          throw new JanusStepError(
            "first-track",
            "first_track_timeout",
            `No media track received within ${Math.round(janusFirstTrackTimeoutMs)}ms`
          );
        }
        return { ok: true, error: "" };
      } catch (err: unknown) {
        const normalizedError =
          err instanceof JanusStepError
            ? err
            : new JanusStepError(
                "first-track",
                "janus_ws_negotiation_failed",
                err instanceof Error ? err.message : "Janus WS negotiation failed"
              );
        if (!(err instanceof JanusStepError)) {
          pushJanusDiagnostic(normalizedError.step, normalizedError.code, normalizedError.message, false);
        }
        closeTransport();
        return {
          ok: false,
          error: `${normalizedError.code} [${normalizedError.step}] ${normalizedError.message}`
        };
      }
    };

    const startJanusTransport = async (): Promise<{ ok: boolean; error: string }> => {
      if (resolvedJanusWsUrl) {
        return startJanusTransportWs();
      }
      janusTransportRef.current = "http";
      return startJanusTransportHttp();
    };

    const startWsJpegTransport = async (): Promise<{ ok: boolean; error: string }> => {
      if (typeof WebSocket === "undefined" || !resolvedWsJpegUrl) {
        return { ok: false, error: "Browser WebSocket unavailable" };
      }
      try {
        const connected = await new Promise<boolean>((resolve) => {
          const socket = new WebSocket(resolvedWsJpegUrl);
          wsJpegSocketRef.current = socket;
          socket.binaryType = "arraybuffer";
          let receivedFrame = false;
          let lastRenderedAt = 0;
          const timeout = window.setTimeout(() => {
            try {
              socket.close();
            } catch {
              // Ignore close failures on timeout.
            }
            resolve(false);
          }, 7000);

          socket.onmessage = (event) => {
            const now = performance.now();
            if (receivedFrame && wsJpegRenderMinIntervalMs > 0 && now - lastRenderedAt < wsJpegRenderMinIntervalMs) {
              return;
            }
            lastRenderedAt = now;
            const blob = event.data instanceof Blob ? event.data : new Blob([event.data], { type: "image/jpeg" });
            const nextUrl = URL.createObjectURL(blob);
            const previousUrl = wsJpegObjectUrlRef.current;
            wsJpegObjectUrlRef.current = nextUrl;
            showMjpeg(nextUrl);
            if (previousUrl) {
              URL.revokeObjectURL(previousUrl);
            }
            if (!receivedFrame) {
              receivedFrame = true;
              window.clearTimeout(timeout);
              transportModeRef.current = "wsjpeg";
              setTransportMode("wsjpeg");
              setTransportError(null);
              resolve(true);
            }
          };
          socket.onerror = () => {
            if (!receivedFrame) {
              window.clearTimeout(timeout);
              resolve(false);
            }
          };
          socket.onclose = () => {
            if (!receivedFrame) {
              window.clearTimeout(timeout);
              resolve(false);
            }
          };
        });
        if (!connected) {
          closeTransport();
          return { ok: false, error: "WS-JPEG stream did not start" };
        }
        return { ok: true, error: "" };
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "WS-JPEG negotiation failed";
        closeTransport();
        return { ok: false, error: message };
      }
    };

    const attemptTransport = async (candidate: TransportCandidate): Promise<{ ok: boolean; error: string }> => {
      if (candidate === "janus") {
        if (disableWebRtc) {
          return { ok: false, error: "WebRTC disabled by VITE_DISABLE_WEBRTC=true" };
        }
        if (disableJanus) {
          return { ok: false, error: "Janus disabled by VITE_DISABLE_JANUS=true" };
        }
        return startJanusTransport();
      }
      if (disableBackendVideo) {
        return { ok: false, error: "Backend video disabled by VITE_DISABLE_BACKEND_VIDEO=true" };
      }
      return startWsJpegTransport();
    };

    const bootstrapVideo = async () => {
      setTransportError(null);
      transportModeRef.current = "connecting";
      setTransportMode("connecting");

      const failures: string[] = [];
      if (!effectiveTransportPlan.length) {
        failures.push("No transports configured");
      }

      for (const candidate of effectiveTransportPlan) {
        const result = await attemptTransport(candidate);
        if (cancelled) {
          return;
        }
        if (result.ok) {
          return;
        }
        failures.push(`${transportLabel(candidate)}: ${result.error}`);
      }

      transportModeRef.current = "error";
      setTransportMode("error");
      const details = failures.join(" | ");
      setTransportError(`Plan ${configuredTransportPlanLabel} | ${details}`);
    };

    bootstrapVideo();

    return () => {
      cancelled = true;
      closeTransport();
    };
  }, [
    active,
    effectiveTransportPlan,
    configuredTransportPlanLabel,
    disableBackendVideo,
    disableJanus,
    disableWebRtc,
    janusFirstTrackTimeoutMs,
    resolvedJanusHttpUrl,
    resolvedJanusWsUrl,
    resolvedJanusMountpoint,
    resolvedWsJpegUrl,
    wsJpegRenderMinIntervalMs
  ]);

  useEffect(() => {
    if (!active) {
      setFaceDetectorUnavailable(false);
      return;
    }
    let cancelled = false;
    const healthUrl = resolveApiUrl("/api/health");

    const pollHealth = async () => {
      try {
        const response = await fetch(healthUrl, { method: "GET", headers: authHeaders });
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
  }, [active, authHeaders]);

  useEffect(() => {
    if (!active) {
      if (rosterFlushTimerRef.current !== null) {
        window.clearTimeout(rosterFlushTimerRef.current);
        rosterFlushTimerRef.current = null;
      }
      latestMetadataRef.current = null;
      drawCacheRef.current = { tracksKey: "", sizeKey: "", focusedTrackId: null };
      rosterEmitRef.current = { key: "", at: 0 };
      rosterPendingRef.current = null;
      metadataLagSamplesRef.current = [];
      overlayIntervalsRef.current = [];
      lastOverlayDrawAtRef.current = null;
      janusConnectStartedAtRef.current = null;
      janusTtffMsRef.current = null;
      onTrackFocus?.(null);
      onRosterChange?.([]);
      requestDrawRef.current();
    }
  }, [active, onTrackFocus, onRosterChange]);

  useEffect(() => {
    return () => {
      if (rosterFlushTimerRef.current !== null) {
        window.clearTimeout(rosterFlushTimerRef.current);
        rosterFlushTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const updateSize = (width: number, height: number) => {
      const nextWidth = Math.max(1, Math.round(width));
      const nextHeight = Math.max(1, Math.round(height));
      const prev = containerSizeRef.current;
      if (prev.width === nextWidth && prev.height === nextHeight) {
        return;
      }
      containerSizeRef.current = { width: nextWidth, height: nextHeight };
      requestDrawRef.current();
    };
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.target !== container) {
          continue;
        }
        updateSize(entry.contentRect.width, entry.contentRect.height);
      }
    });
    observer.observe(container);
    const initialRect = container.getBoundingClientRect();
    updateSize(initialRect.width, initialRect.height);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const mjpeg = mjpegRef.current;
    const handleMediaResize = () => {
      requestDrawRef.current();
    };
    video?.addEventListener("loadedmetadata", handleMediaResize);
    video?.addEventListener("loadeddata", handleMediaResize);
    video?.addEventListener("resize", handleMediaResize);
    mjpeg?.addEventListener("load", handleMediaResize);
    return () => {
      video?.removeEventListener("loadedmetadata", handleMediaResize);
      video?.removeEventListener("loadeddata", handleMediaResize);
      video?.removeEventListener("resize", handleMediaResize);
      mjpeg?.removeEventListener("load", handleMediaResize);
    };
  }, []);

  useEffect(() => {
    const drawNow = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }
      const cssWidth = Math.max(1, containerSizeRef.current.width);
      const cssHeight = Math.max(1, containerSizeRef.current.height);
      let nativeW = 0;
      let nativeH = 0;
      const video = videoRef.current;
      if (video) {
        nativeW = video.videoWidth;
        nativeH = video.videoHeight;
      }
      if ((!nativeW || !nativeH) && mjpegRef.current) {
        nativeW = mjpegRef.current.naturalWidth;
        nativeH = mjpegRef.current.naturalHeight;
      }

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
        return;
      }
      ctx.setTransform(canvas.width / cssWidth, 0, 0, canvas.height / cssHeight, 0, 0);
      ctx.imageSmoothingEnabled = true;
      metricsRef.current = { cssWidth, cssHeight, sourceWidth, sourceHeight };

      const tracksKey = computeTracksKey(latest);
      const sizeKey = `${cssWidth}x${cssHeight}:${sourceWidth}x${sourceHeight}:${pixelPerfect ? "pp" : "fit"}`;
      const cache = drawCacheRef.current;
      const shouldDraw =
        cache.tracksKey !== tracksKey ||
        cache.sizeKey !== sizeKey ||
        cache.focusedTrackId !== focusedTrackId;
      if (!shouldDraw) {
        return;
      }
      cache.tracksKey = tracksKey;
      cache.sizeKey = sizeKey;
      cache.focusedTrackId = focusedTrackId;

      ctx.clearRect(0, 0, cssWidth, cssHeight);
      projectedTracksRef.current = [];
      if (!active || !latest || sourceWidth < 2 || sourceHeight < 2) {
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
        const x = offsetX + x1 * scaleX;
        const y = offsetY + y1 * scaleY;
        const w = bw * scaleX;
        const h = bh * scaleY;
        if (w <= 1 || h <= 1) {
          continue;
        }

        const isFocused = focusedTrackId !== null && focusedTrackId === track.track_id;
        const isMuted = Boolean(track.muted);
        const ageRatio = Math.max(0, Math.min(1, Number(track.age_ratio ?? 0)));
        const freshness = 1 - ageRatio;
        const strokeAlpha = isFocused ? 1 : 0.35 + freshness * 0.6;
        const fillAlpha = isFocused ? 0.13 : 0.05 + freshness * 0.11;
        const accent = isFocused
          ? "rgba(255, 198, 89, 1)"
          : isMuted
            ? `rgba(148, 163, 184, ${Math.max(0.2, Math.min(1, strokeAlpha)).toFixed(3)})`
            : `rgba(124, 92, 255, ${Math.max(0.2, Math.min(1, strokeAlpha)).toFixed(3)})`;
        const fill = isFocused
          ? "rgba(255, 198, 89, 0.13)"
          : isMuted
            ? `rgba(148, 163, 184, ${Math.max(0.03, Math.min(0.25, fillAlpha)).toFixed(3)})`
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

      const nowMs = performance.now();
      if (lastOverlayDrawAtRef.current !== null && nowMs > lastOverlayDrawAtRef.current) {
        const deltaMs = nowMs - lastOverlayDrawAtRef.current;
        const intervals = overlayIntervalsRef.current;
        intervals.push(deltaMs);
        if (intervals.length > 256) {
          intervals.splice(0, intervals.length - 256);
        }
      }
      lastOverlayDrawAtRef.current = nowMs;
    };

    const requestDraw = () => {
      if (drawScheduledRef.current) {
        return;
      }
      drawScheduledRef.current = true;
      drawRafRef.current = window.requestAnimationFrame(() => {
        drawScheduledRef.current = false;
        drawRafRef.current = null;
        drawNow();
      });
    };
    requestDrawRef.current = requestDraw;
    requestDraw();

    return () => {
      requestDrawRef.current = () => undefined;
      drawScheduledRef.current = false;
      if (drawRafRef.current !== null) {
        window.cancelAnimationFrame(drawRafRef.current);
        drawRafRef.current = null;
      }
    };
  }, [active, focusedTrackId, pixelPerfect]);

  useEffect(() => {
    if (!active) {
      return;
    }
    const perfUrl = resolveApiUrl("/api/perf/client");
    const timer = window.setInterval(() => {
      const intervals = overlayIntervalsRef.current;
      const metadataLagP95 = percentile(metadataLagSamplesRef.current, 95);
      const overlayAvgMs = intervals.length ? intervals.reduce((sum, value) => sum + value, 0) / intervals.length : null;
      const overlayFps = overlayAvgMs && overlayAvgMs > 0 ? 1000 / overlayAvgMs : null;
      const overlayJankRatio = intervals.length
        ? intervals.filter((value) => value > 50).length / intervals.length
        : null;
      const janusTtffMs = janusTtffMsRef.current;
      const payload: Record<string, number> = {};
      if (metadataLagP95 !== null && Number.isFinite(metadataLagP95)) {
        payload.metadata_lag_ms_p95 = metadataLagP95;
      }
      if (overlayFps !== null && Number.isFinite(overlayFps)) {
        payload.overlay_fps = overlayFps;
      }
      if (overlayJankRatio !== null && Number.isFinite(overlayJankRatio)) {
        payload.overlay_jank_ratio = overlayJankRatio;
      }
      if (janusTtffMs !== null && Number.isFinite(janusTtffMs)) {
        payload.janus_ttff_ms = janusTtffMs;
      }
      if (Object.keys(payload).length === 0 || perfPostInflightRef.current) {
        return;
      }
      perfPostInflightRef.current = true;
      const headers = withApiKeyHeaders(authHeaders);
      headers.set("Content-Type", "application/json");
      void fetch(perfUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(payload)
      }).finally(() => {
        perfPostInflightRef.current = false;
      });
    }, 2000);
    return () => {
      window.clearInterval(timer);
    };
  }, [active, authHeaders]);

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
        <span>Plan: {configuredTransportPlanLabel}</span>
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
