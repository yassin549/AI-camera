import { useEffect, useRef, useState, type MutableRefObject } from "react";
import type { MetadataPayload, TrackPayload } from "../api/client";

export type RealtimeStatus = "idle" | "connecting" | "open" | "closed" | "error";

interface UseRealtimeOptions {
  url: string;
  enabled?: boolean;
  onMessage?: (payload: MetadataPayload) => void;
}

interface UseRealtimeResult {
  status: RealtimeStatus;
  error: string | null;
  retryAttempt: number;
  latestRef: MutableRefObject<MetadataPayload | null>;
}

const RECONNECT_DELAYS_MS = [100, 250, 500, 1000, 2000];

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
  const trackId = Number(source.track_id ?? asRecord.id ?? -1);
  const identityIdRaw = source.identity_id ?? asRecord.identityId;
  const score = Number(asRecord.score ?? asRecord.confidence ?? asRecord.last_score ?? 0);
  const label =
    (typeof source.label === "string" && source.label.trim()) ||
    (identityIdRaw !== null && identityIdRaw !== undefined
      ? `ID:${identityIdRaw} (${score.toFixed(2)})`
      : `Track ${trackId}`);

  return {
    track_id: trackId,
    bbox: [x1, y1, x2, y2],
    identity_id:
      identityIdRaw === null || identityIdRaw === undefined
        ? null
        : Number(identityIdRaw),
    label: String(label),
    modality: String(source.modality ?? "none"),
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

function parseMessage(raw: string): MetadataPayload | null {
  try {
    const parsedAny = JSON.parse(raw) as unknown;
    if (Array.isArray(parsedAny)) {
      const tracks = parsedAny.map(coerceTrack).filter((v): v is TrackPayload => v !== null);
      return {
        frame_id: -1,
        timestamp: new Date().toISOString(),
        source_width: 0,
        source_height: 0,
        tracks
      };
    }
    const parsed = parsedAny as Partial<MetadataPayload> & {
      tracks?: unknown;
      payload?: { tracks?: unknown; frame_id?: number; timestamp?: string; source_width?: number; source_height?: number };
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
      source_width: Number(parsed.payload?.source_width ?? parsed.source_width ?? 0),
      source_height: Number(parsed.payload?.source_height ?? parsed.source_height ?? 0),
      tracks
    };
  } catch {
    return null;
  }
}

export function useRealtime(options: UseRealtimeOptions): UseRealtimeResult {
  const { url, enabled = true, onMessage } = options;
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
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

    const scheduleReconnect = () => {
      if (cancelled) {
        return;
      }
      const idx = Math.min(attemptRef.current, RECONNECT_DELAYS_MS.length - 1);
      const delay = RECONNECT_DELAYS_MS[idx];
      attemptRef.current += 1;
      setRetryAttempt(attemptRef.current);
      timerRef.current = window.setTimeout(connect, delay);
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
        attemptRef.current = 0;
        setRetryAttempt(0);
        setStatus("open");
        setError(null);
      };

      wsRef.current.onmessage = (event) => {
        const parsed = parseMessage(String(event.data));
        if (!parsed) {
          return;
        }
        latestRef.current = parsed;
        onMessageRef.current?.(parsed);
      };

      wsRef.current.onerror = () => {
        setStatus("error");
        setError("WebSocket transport error");
      };

      wsRef.current.onclose = () => {
        setStatus("closed");
        scheduleReconnect();
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
      cleanupSocket();
    };
  }, [enabled, url]);

  return { status, error, retryAttempt, latestRef };
}
