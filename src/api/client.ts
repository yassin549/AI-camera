import { API } from "../config";

export interface IdentityStats {
  frequency?: number;
  sightings?: number;
}

export interface IdentitySummary {
  id: string;
  name?: string;
  display_name?: string;
  first_seen: string;
  last_seen: string;
  is_muted?: boolean;
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

export interface TrackPayload {
  track_id: number;
  bbox: [number, number, number, number];
  identity_id: number | null;
  label: string;
  modality: "face" | "body" | "none" | string;
  muted?: boolean;
  age_ratio?: number;
  age_frames?: number;
  thumb?: string;
}

export interface MetadataPayload {
  frame_id: number;
  timestamp: string;
  capture_ts?: string | null;
  capture_ts_unix?: number | null;
  metadata_lag_ms?: number | null;
  source_width: number;
  source_height: number;
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

function getApiKey(): string | null {
  if (!API.API_KEY) {
    return null;
  }
  const normalized = String(API.API_KEY).trim();
  return normalized.length > 0 ? normalized : null;
}

function isNgrokFreeDomain(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host.endsWith(".ngrok-free.app") || host.endsWith(".ngrok-free.dev");
  } catch {
    return false;
  }
}

function appendQueryParam(url: string, key: string, value: string): string {
  try {
    const parsed = new URL(url);
    if (!parsed.searchParams.has(key)) {
      parsed.searchParams.set(key, value);
    }
    return parsed.toString();
  } catch {
    const [base, hash = ""] = url.split("#", 2);
    const hasQuery = new RegExp(`(?:\\?|&)${key}=`).test(base);
    if (hasQuery) {
      return url;
    }
    const separator = base.includes("?") ? "&" : "?";
    const next = `${base}${separator}${encodeURIComponent(key)}=${encodeURIComponent(value)}`;
    return hash ? `${next}#${hash}` : next;
  }
}

export function withApiKeyQuery(url: string): string {
  if (!url) {
    return url;
  }
  let next = url;
  const key = getApiKey();
  if (key) {
    next = appendQueryParam(next, "api_key", key);
  }
  if (isNgrokFreeDomain(next)) {
    next = appendQueryParam(next, "ngrok-skip-browser-warning", "1");
  }
  return next;
}

export function withApiKeyHeaders(input?: HeadersInit): Headers {
  const headers = new Headers(input ?? {});
  const key = getApiKey();
  if (key && !headers.has("x-api-key")) {
    headers.set("x-api-key", key);
  }
  if (isNgrokFreeDomain(API.REST_BASE) && !headers.has("ngrok-skip-browser-warning")) {
    headers.set("ngrok-skip-browser-warning", "1");
  }
  return headers;
}

function toWsUrl(input: string): string {
  if (!input) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/api/realtime/ws`;
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
      .map((entry) => withApiKeyQuery(toMediaUrl(entry)));
  }
  const single = asString(input);
  return single ? [withApiKeyQuery(toMediaUrl(single))] : [];
}

function normalizeIdentity(raw: unknown): IdentitySummary {
  const source = (raw ?? {}) as Record<string, unknown>;
  const id = asString(source.id ?? source.identity_id ?? source.identityId) ?? "unknown";
  const displayName = asString(source.display_name ?? source.displayName ?? source.name) ?? undefined;
  const firstSeen =
    asString(source.first_seen ?? source.created_ts ?? source.createdAt ?? source.firstSeen) ??
    new Date(0).toISOString();
  const lastSeen =
    asString(source.last_seen ?? source.last_seen_ts ?? source.lastSeen ?? source.updated_at) ?? firstSeen;
  const mutedRaw = source.is_muted ?? source.muted;
  const isMuted =
    typeof mutedRaw === "boolean"
      ? mutedRaw
      : typeof mutedRaw === "number"
        ? mutedRaw !== 0
        : typeof mutedRaw === "string"
          ? ["1", "true", "yes"].includes(mutedRaw.trim().toLowerCase())
          : undefined;

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
  const sampleImages = [
    ...coerceStringArray(source.sample_images),
    ...coerceStringArray(source.sampleImages),
    ...coerceStringArray(source.samples)
  ];
  if (faceSamples.length === 0 && bodySamples.length === 0 && sampleImages.length > 0) {
    faceSamples.push(...sampleImages);
  }

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
    ...(displayName ? { name: displayName } : {}),
    ...(displayName ? { display_name: displayName } : {}),
    first_seen: firstSeen,
    last_seen: lastSeen,
    ...(isMuted !== undefined ? { is_muted: isMuted } : {}),
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
  const headers = withApiKeyHeaders(init?.headers);
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

export async function snapshotIdentity(id: string): Promise<{ ok: true; sample?: string }> {
  return request<{ ok: true; sample?: string }>(`/api/identities/${encodeURIComponent(id)}/snapshot`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export async function muteIdentity(id: string, muted?: boolean): Promise<{ ok: true; muted: boolean }> {
  return request<{ ok: true; muted: boolean }>(`/api/identities/${encodeURIComponent(id)}/mute`, {
    method: "POST",
    body: JSON.stringify(muted === undefined ? {} : { muted })
  });
}

export async function assignTrackIdentity(
  trackId: number,
  identityId: string | number,
  cacheSeconds?: number
): Promise<{ ok: true; track_id: number; identity_id: number; cache_seconds: number }> {
  const body: Record<string, unknown> = {
    identity_id: Number(identityId)
  };
  if (typeof cacheSeconds === "number" && Number.isFinite(cacheSeconds)) {
    body.cache_seconds = cacheSeconds;
  }
  return request<{ ok: true; track_id: number; identity_id: number; cache_seconds: number }>(
    `/api/tracks/${encodeURIComponent(String(trackId))}/assign`,
    {
      method: "POST",
      body: JSON.stringify(body)
    }
  );
}

export function resolveMetadataWsUrl(): string {
  return withApiKeyQuery(toWsUrl(API.WS_METADATA));
}

export function resolveMetadataLatestUrl(): string {
  return withApiKeyQuery(toMediaUrl(API.METADATA_LATEST));
}
