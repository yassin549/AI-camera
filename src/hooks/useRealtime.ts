import { useEffect, useRef, useState, type MutableRefObject } from "react";
import type { MetadataPayload, TrackPayload } from "../api/client";

export type RealtimeStatus = "idle" | "connecting" | "open" | "closed" | "error";

interface UseRealtimeOptions {
  url: string;
  enabled?: boolean;
  onMessage?: (payload: MetadataPayload) => void;
  latestUrl?: string;
  pollIntervalMs?: number;
  staleTimeoutMs?: number;
  headers?: HeadersInit;
}

interface UseRealtimeResult {
  status: RealtimeStatus;
  error: string | null;
  retryAttempt: number;
  latestRef: MutableRefObject<MetadataPayload | null>;
}

const RECONNECT_DELAYS_MS = [100, 250, 500, 1000, 2000];

function payloadSortKey(payload: MetadataPayload): number {
  const frameId = Number(payload.frame_id);
  if (Number.isFinite(frameId) && frameId >= 0) {
    return frameId;
  }
  const ts = Date.parse(String(payload.capture_ts ?? payload.timestamp ?? ""));
  return Number.isFinite(ts) ? ts : 0;
}

function asNullableNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function asNullableBoolean(value: unknown): boolean | null {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    if (value === 0) {
      return false;
    }
    if (value === 1) {
      return true;
    }
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no"].includes(normalized)) {
      return false;
    }
  }
  return null;
}

function coerceTrack(raw: unknown): TrackPayload | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const source = raw as Partial<TrackPayload>;
  const fromBox = Array.isArray(source.bbox) ? source.bbox : null;
  const asRecord = source as Record<string, unknown>;
  const bboxSource = fromBox;
  if (!bboxSource || bboxSource.length !== 4) {
    return null;
  }
  const x1 = Number(bboxSource[0] ?? 0);
  const y1 = Number(bboxSource[1] ?? 0);
  const x2 = Number(bboxSource[2] ?? 0);
  const y2 = Number(bboxSource[3] ?? 0);
  if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) {
    return null;
  }
  if (x2 <= x1 || y2 <= y1) {
    // Runtime assertion: invalid box contract should be visible during operation.
    console.error("Invalid bbox(xyxy) received from metadata stream", { bbox: [x1, y1, x2, y2], raw });
    return null;
  }
  const trackIdRaw = asNullableNumber(source.track_id ?? asRecord.id);
  if (trackIdRaw === null || trackIdRaw < 0) {
    return null;
  }
  const trackId = trackIdRaw;
  const identityIdRaw = source.identity_id ?? asRecord.identityId;
  const identityId = asNullableNumber(identityIdRaw);
  const score = Number(asRecord.score ?? asRecord.confidence ?? asRecord.last_score ?? 0);
  const muted = asNullableBoolean(asRecord.muted);
  const label =
    (typeof source.label === "string" && source.label.trim()) ||
    (identityId !== null
      ? `ID:${identityId} (${score.toFixed(2)})`
      : `Track ${trackId}`);

  return {
    track_id: trackId,
    bbox: [x1, y1, x2, y2],
    identity_id: identityId,
    label: String(label),
    modality: String(source.modality ?? "none"),
    muted: muted ?? undefined,
    age_ratio: Number(asRecord.age_ratio ?? 0),
    age_frames: Number(asRecord.age_frames ?? 0),
    thumb:
      typeof source.thumb === "string"
        ? source.thumb
        : typeof asRecord.face_sample_path === "string"
          ? String(asRecord.face_sample_path)
          : undefined
  };
}

function normalizePayload(parsedAny: unknown): MetadataPayload | null {
  try {
    if (Array.isArray(parsedAny)) {
      const tracks = parsedAny.map(coerceTrack).filter((v): v is TrackPayload => v !== null);
      return {
        frame_id: -1,
        timestamp: new Date().toISOString(),
        capture_ts: null,
        capture_ts_unix: null,
        metadata_lag_ms: null,
        source_width: 0,
        source_height: 0,
        tracks
      };
    }
    const parsed = parsedAny as Partial<MetadataPayload> & {
      tracks?: unknown;
      payload?: {
        tracks?: unknown;
        frame_id?: number;
        timestamp?: string;
        capture_ts?: string;
        capture_ts_unix?: number;
        metadata_lag_ms?: number;
        source_width?: number;
        source_height?: number;
      };
    };
    const tracksInput = parsed.payload?.tracks ?? parsed.tracks;
    const rawTracks = Array.isArray(tracksInput)
      ? tracksInput
      : tracksInput && typeof tracksInput === "object"
        ? Object.values(tracksInput as Record<string, unknown>)
        : null;
    if (!rawTracks) {
      return null;
    }
    const tracks = rawTracks.map(coerceTrack).filter((v): v is TrackPayload => v !== null);
    return {
      frame_id: Number(parsed.payload?.frame_id ?? parsed.frame_id ?? -1),
      timestamp: String(parsed.payload?.timestamp ?? parsed.timestamp ?? new Date().toISOString()),
      capture_ts: (parsed.payload as Record<string, unknown> | undefined)?.capture_ts
        ? String((parsed.payload as Record<string, unknown>).capture_ts)
        : parsed.capture_ts
          ? String(parsed.capture_ts)
          : null,
      capture_ts_unix: asNullableNumber(
        (parsed.payload as Record<string, unknown> | undefined)?.capture_ts_unix ??
          (parsed as Record<string, unknown>).capture_ts_unix
      ),
      metadata_lag_ms: asNullableNumber(
        (parsed.payload as Record<string, unknown> | undefined)?.metadata_lag_ms ??
          (parsed as Record<string, unknown>).metadata_lag_ms
      ),
      source_width: Number(parsed.payload?.source_width ?? parsed.source_width ?? 0),
      source_height: Number(parsed.payload?.source_height ?? parsed.source_height ?? 0),
      tracks
    };
  } catch {
    return null;
  }
}

function parseMessage(raw: string): MetadataPayload | null {
  try {
    return normalizePayload(JSON.parse(raw) as unknown);
  } catch {
    return null;
  }
}

export function useRealtime(options: UseRealtimeOptions): UseRealtimeResult {
  const { url, enabled = true, onMessage, latestUrl, pollIntervalMs = 250, staleTimeoutMs, headers } = options;
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const parseTimerRef = useRef<number | null>(null);
  const pendingRawRef = useRef<string | null>(null);
  const lastActivityAtRef = useRef(0);
  const lastPayloadAtRef = useRef(0);
  const attemptRef = useRef(0);
  const latestRef = useRef<MetadataPayload | null>(null);
  const onMessageRef = useRef(onMessage);
  const [status, setStatus] = useState<RealtimeStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [retryAttempt, setRetryAttempt] = useState(0);

  onMessageRef.current = onMessage;

  useEffect(() => {
    if (!enabled) {
      setStatus("idle");
      return;
    }

    let cancelled = false;

    const cleanupSocket = () => {
      const current = wsRef.current;
      if (current) {
        current.onopen = null;
        current.onclose = null;
        current.onerror = null;
        current.onmessage = null;
        current.close();
        wsRef.current = null;
      }
    };

    const clearTimers = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (pollTimerRef.current !== null) {
        window.clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      if (parseTimerRef.current !== null) {
        window.clearTimeout(parseTimerRef.current);
        parseTimerRef.current = null;
      }
      pendingRawRef.current = null;
    };

    const publishPayload = (payload: MetadataPayload) => {
      const previous = latestRef.current;
      if (previous) {
        const nextKey = payloadSortKey(payload);
        const prevKey = payloadSortKey(previous);
        if (nextKey < prevKey) {
          return;
        }
      }
      lastPayloadAtRef.current = Date.now();
      latestRef.current = payload;
      onMessageRef.current?.(payload);
    };

    const fetchLatest = async () => {
      if (!latestUrl || cancelled) {
        return;
      }
      try {
        const response = await fetch(latestUrl, { method: "GET", headers });
        if (!response.ok) {
          return;
        }
        const parsed = normalizePayload((await response.json()) as unknown);
        if (parsed && !cancelled) {
          lastActivityAtRef.current = Date.now();
          publishPayload(parsed);
        }
      } catch {
        // Keep websocket reconnect loop as primary transport.
      }
    };

    const scheduleMetadataPoll = () => {
      if (!latestUrl || cancelled) {
        return;
      }
      const interval = Math.max(80, Number.isFinite(pollIntervalMs) ? Math.round(pollIntervalMs) : 250);
      const staleTimeoutCandidate = typeof staleTimeoutMs === "number" ? staleTimeoutMs : NaN;
      const staleSoftMs = Math.max(1000, Number.isFinite(staleTimeoutCandidate) ? Math.round(staleTimeoutCandidate) : interval * 6);
      const staleHardMs = Math.max(staleSoftMs * 4, 8000);
      const tick = () => {
        if (cancelled) {
          return;
        }
        const socket = wsRef.current;
        const isOpen = socket?.readyState === WebSocket.OPEN;
        const now = Date.now();
        const idleFor = lastPayloadAtRef.current > 0 ? now - lastPayloadAtRef.current : Number.POSITIVE_INFINITY;
        const inactiveFor = lastActivityAtRef.current > 0 ? now - lastActivityAtRef.current : Number.POSITIVE_INFINITY;

        if (!isOpen || idleFor >= staleSoftMs) {
          void fetchLatest();
        }
        if (isOpen && inactiveFor >= staleHardMs) {
          try {
            socket?.close();
          } catch {
            // Ignore close failures; reconnect loop will continue.
          }
        }
        pollTimerRef.current = window.setTimeout(tick, interval);
      };
      pollTimerRef.current = window.setTimeout(tick, 0);
    };

    const scheduleReconnect = () => {
      if (cancelled) {
        return;
      }
      const idx = Math.min(attemptRef.current, RECONNECT_DELAYS_MS.length - 1);
      const delay = RECONNECT_DELAYS_MS[idx];
      attemptRef.current += 1;
      setRetryAttempt(attemptRef.current);
      reconnectTimerRef.current = window.setTimeout(connect, delay);
    };

    const connect = () => {
      if (cancelled) {
        return;
      }
      cleanupSocket();
      setStatus("connecting");
      try {
        wsRef.current = new WebSocket(url);
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : "WebSocket initialization failed");
        scheduleReconnect();
        return;
      }

      wsRef.current.onopen = () => {
        const now = Date.now();
        lastActivityAtRef.current = now;
        if (lastPayloadAtRef.current <= 0) {
          lastPayloadAtRef.current = now;
        }
        attemptRef.current = 0;
        setRetryAttempt(0);
        setStatus("open");
        setError(null);
      };

      wsRef.current.onmessage = (event) => {
        lastActivityAtRef.current = Date.now();
        pendingRawRef.current = String(event.data);
        if (parseTimerRef.current !== null) {
          return;
        }
        parseTimerRef.current = window.setTimeout(() => {
          parseTimerRef.current = null;
          const raw = pendingRawRef.current;
          pendingRawRef.current = null;
          if (!raw) {
            return;
          }
          const parsed = parseMessage(raw);
          if (!parsed) {
            return;
          }
          publishPayload(parsed);
        }, 0);
      };

      wsRef.current.onerror = () => {
        setStatus("error");
        setError("WebSocket transport error");
        try {
          wsRef.current?.close();
        } catch {
          // Ignore close failures and rely on reconnect backoff.
        }
      };

      wsRef.current.onclose = () => {
        setStatus("closed");
        scheduleReconnect();
      };
    };

    scheduleMetadataPoll();
    void fetchLatest();
    connect();

    return () => {
      cancelled = true;
      clearTimers();
      cleanupSocket();
    };
  }, [enabled, headers, latestUrl, pollIntervalMs, staleTimeoutMs, url]);

  return { status, error, retryAttempt, latestRef };
}
