import { API } from "../config";

export interface IdentityStats {
  frequency?: number;
  sightings?: number;
}

export interface IdentitySummary {
  id: string;
  first_seen: string;
  last_seen: string;
  face_samples: string[];
  body_samples: string[];
  stats?: IdentityStats;
}

export interface IdentityTimelineEvent {
  at: string;
  note: string;
}

export interface IdentityDetailData extends IdentitySummary {
  aliases?: string[];
  timeline?: IdentityTimelineEvent[];
  samples?: string[];
}

export interface WebRtcOfferPayload {
  sdp: string;
  type: "offer";
}

export interface WebRtcAnswerPayload {
  sdp: string;
  type: "answer";
}

export interface TrackPayload {
  track_id: number;
  bbox: [number, number, number, number];
  identity_id: number | null;
  label: string;
  modality: "face" | "body" | "none" | string;
  thumb?: string;
}

export interface MetadataPayload {
  frame_id: number;
  timestamp: string;
  tracks: TrackPayload[];
}

export class HttpError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function joinUrl(base: string, path: string): string {
  if (/^https?:\/\//i.test(path) || /^wss?:\/\//i.test(path)) {
    return path;
  }
  if (!base) {
    return path;
  }
  const cleanBase = base.replace(/\/+$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${cleanBase}${cleanPath}`;
}

function toWsUrl(input: string): string {
  if (!input) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/metadata`;
  }
  if (/^wss?:\/\//i.test(input)) {
    return input;
  }
  if (/^https?:\/\//i.test(input)) {
    return input.replace(/^http/i, "ws");
  }
  if (input.startsWith("/")) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}${input}`;
  }
  return input;
}

function toMediaUrl(input: string): string {
  if (!input) {
    return input;
  }
  if (/^(https?:)?\/\//i.test(input) || input.startsWith("data:") || input.startsWith("blob:")) {
    return input;
  }
  return joinUrl(API.REST_BASE, input.startsWith("/") ? input : `/${input}`);
}

function asString(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return null;
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function coerceStringArray(input: unknown): string[] {
  if (Array.isArray(input)) {
    return input
      .map((entry) => asString(entry))
      .filter((entry): entry is string => Boolean(entry))
      .map((entry) => toMediaUrl(entry));
  }
  const single = asString(input);
  return single ? [toMediaUrl(single)] : [];
}

function normalizeIdentity(raw: unknown): IdentitySummary {
  const source = (raw ?? {}) as Record<string, unknown>;
  const id = asString(source.id ?? source.identity_id ?? source.identityId) ?? "unknown";
  const firstSeen =
    asString(source.first_seen ?? source.created_ts ?? source.createdAt ?? source.firstSeen) ??
    new Date(0).toISOString();
  const lastSeen =
    asString(source.last_seen ?? source.last_seen_ts ?? source.lastSeen ?? source.updated_at) ?? firstSeen;

  const faceSamples = [
    ...coerceStringArray(source.face_samples),
    ...coerceStringArray(source.face_sample_path),
    ...coerceStringArray(source.face_sample)
  ];
  const bodySamples = [
    ...coerceStringArray(source.body_samples),
    ...coerceStringArray(source.body_sample_path),
    ...coerceStringArray(source.body_sample)
  ];

  const statsSource = (source.stats ?? {}) as Record<string, unknown>;
  const frequency = asNumber(statsSource.frequency ?? source.frequency ?? source.count ?? source.sightings);
  const sightings = asNumber(statsSource.sightings ?? source.sightings);
  const stats: IdentityStats | undefined =
    frequency !== null || sightings !== null
      ? {
          ...(frequency !== null ? { frequency } : {}),
          ...(sightings !== null ? { sightings } : {})
        }
      : undefined;

  return {
    id,
    first_seen: firstSeen,
    last_seen: lastSeen,
    face_samples: [...new Set(faceSamples)],
    body_samples: [...new Set(bodySamples)],
    stats
  };
}

function normalizeIdentityList(payload: unknown): IdentitySummary[] {
  if (Array.isArray(payload)) {
    return payload.map(normalizeIdentity);
  }
  if (payload && typeof payload === "object") {
    const source = payload as Record<string, unknown>;
    if (Array.isArray(source.identities)) {
      return source.identities.map(normalizeIdentity);
    }
    return Object.values(source).map(normalizeIdentity);
  }
  return [];
}

function normalizeIdentityDetail(payload: unknown): IdentityDetailData {
  const source = (payload ?? {}) as Record<string, unknown>;
  const base = normalizeIdentity(source);
  const aliases = coerceStringArray(source.aliases);
  const timeline = Array.isArray(source.timeline)
    ? source.timeline
        .map((item) => {
          const event = (item ?? {}) as Record<string, unknown>;
          const at = asString(event.at ?? event.timestamp);
          const note = asString(event.note ?? event.label ?? event.type);
          if (!at || !note) {
            return null;
          }
          return { at, note };
        })
        .filter((item): item is IdentityTimelineEvent => item !== null)
    : undefined;
  const samples = coerceStringArray(source.samples);

  return {
    ...base,
    ...(aliases.length > 0 ? { aliases } : {}),
    ...(timeline && timeline.length > 0 ? { timeline } : {}),
    ...(samples.length > 0 ? { samples } : {})
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  const hasBody = Boolean(init?.body);
  if (hasBody && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(joinUrl(API.REST_BASE, path), {
    ...init,
    headers
  });
  if (!response.ok) {
    let details = "";
    try {
      details = await response.text();
    } catch {
      details = "";
    }
    throw new HttpError(
      response.status,
      details || `Request failed (${response.status}) for ${path}`
    );
  }
  if (response.status === 204) {
    return undefined as T;
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }

  return (await response.text()) as T;
}

export async function getIdentities(): Promise<IdentitySummary[]> {
  const payload = await request<unknown>("/api/identities");
  return normalizeIdentityList(payload);
}

export async function getIdentityById(id: string): Promise<IdentityDetailData> {
  const encoded = encodeURIComponent(id);
  try {
    const payload = await request<unknown>(`/api/identities/${encoded}`);
    return normalizeIdentityDetail(payload);
  } catch (error) {
    if (!(error instanceof HttpError) || error.status !== 404) {
      throw error;
    }
    const identities = await getIdentities();
    const fallback = identities.find((identity) => identity.id === id);
    if (!fallback) {
      throw error;
    }
    return {
      ...fallback,
      timeline: [
        { at: fallback.first_seen, note: "First observed" },
        { at: fallback.last_seen, note: "Most recent observation" }
      ],
      samples: [...fallback.face_samples, ...fallback.body_samples]
    };
  }
}

export async function postWebRtcOffer(
  offer: WebRtcOfferPayload,
  path = API.WEBRTC_OFFER
): Promise<WebRtcAnswerPayload> {
  const response = await fetch(joinUrl(API.REST_BASE, path), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(offer)
  });
  if (!response.ok) {
    let details = "";
    try {
      const contentType = response.headers.get("content-type") ?? "";
      if (contentType.includes("application/json")) {
        const payload = (await response.json()) as { detail?: string };
        details = payload.detail ? String(payload.detail) : "";
      } else {
        details = await response.text();
      }
    } catch {
      details = "";
    }
    const message = details
      ? `WebRTC offer failed (${response.status}): ${details}`
      : `WebRTC offer failed (${response.status})`;
    throw new HttpError(response.status, message);
  }
  return (await response.json()) as WebRtcAnswerPayload;
}

export async function renameIdentity(id: string, name: string): Promise<{ ok: true }> {
  return request<{ ok: true }>(`/api/identities/${encodeURIComponent(id)}/rename`, {
    method: "POST",
    body: JSON.stringify({ name })
  });
}

export async function mergeIdentity(
  id: string,
  targetId: string
): Promise<{ ok: true; merged_into: string }> {
  return request<{ ok: true; merged_into: string }>(
    `/api/identities/${encodeURIComponent(id)}/merge`,
    {
      method: "POST",
      body: JSON.stringify({ target_id: targetId })
    }
  );
}

export async function deleteIdentity(id: string): Promise<{ ok: true }> {
  return request<{ ok: true }>(`/api/identities/${encodeURIComponent(id)}`, {
    method: "DELETE"
  });
}

export async function snapshotIdentity(id: string): Promise<{ ok: true }> {
  return request<{ ok: true }>(`/api/identities/${encodeURIComponent(id)}/snapshot`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export async function muteIdentity(id: string): Promise<{ ok: true }> {
  return request<{ ok: true }>(`/api/identities/${encodeURIComponent(id)}/mute`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function resolveMetadataWsUrl(): string {
  return toWsUrl(API.WS_METADATA);
}
