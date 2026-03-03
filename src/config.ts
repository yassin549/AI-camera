type EnvMap = Record<string, string | undefined>;
type PrimaryVideoTransport = "janus";
type VideoFallbackTransport = "wsjpeg" | "none";
type VideoTransportCandidate = "janus" | "wsjpeg";

const appEnv = typeof __APP_ENV__ !== "undefined" ? (__APP_ENV__ as EnvMap) : ({} as EnvMap);
const processEnv = (typeof process !== "undefined" ? (process.env as EnvMap | undefined) : undefined) ?? {};

function readEnv(...keys: string[]): string | undefined {
  for (const key of keys) {
    const fromAppEnv = appEnv[key];
    if (fromAppEnv !== undefined && fromAppEnv !== "") {
      return fromAppEnv;
    }
    const fromProcess = processEnv[key];
    if (fromProcess !== undefined && fromProcess !== "") {
      return fromProcess;
    }
  }
  return undefined;
}

function asBool(value: string | undefined): boolean {
  if (!value) {
    return false;
  }
  const normalized = value.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes";
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function asPositiveInt(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return parsed;
}

function normalizePrimaryTransport(
  _value: string | undefined,
  fallback: PrimaryVideoTransport
): PrimaryVideoTransport {
  return fallback;
}

function normalizeFallbackTransport(value: string | undefined, fallback: VideoFallbackTransport): VideoFallbackTransport {
  const normalized = (value ?? "").trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (normalized === "none" || normalized === "off" || normalized === "disabled") {
    return "none";
  }
  if (normalized === "wsjpeg") {
    return "wsjpeg";
  }
  return fallback;
}

function buildTransportPlan(
  primary: PrimaryVideoTransport,
  fallback: VideoFallbackTransport
): readonly VideoTransportCandidate[] {
  const ordered: VideoTransportCandidate[] = [primary];
  if (fallback === "wsjpeg") {
    ordered.push("wsjpeg");
  }
  return Object.freeze(ordered);
}

const defaultRestBase = "http://localhost:8080";
const restBase = readEnv("VITE_API_BASE", "REACT_APP_API_URL") ?? defaultRestBase;
const normalizedRestBase = trimTrailingSlash(restBase || defaultRestBase);
const wsMetadata =
  readEnv("VITE_WS_METADATA_URL", "REACT_APP_WS_URL") ??
  `${normalizedRestBase}/api/realtime/ws`;
const janusHttp =
  readEnv("VITE_JANUS_HTTP_URL", "REACT_APP_JANUS_HTTP_URL") ??
  `${normalizedRestBase}/janus`;
const janusMountpoint = Number(readEnv("VITE_JANUS_MOUNTPOINT", "REACT_APP_JANUS_MOUNTPOINT") ?? "1");
const useMock = asBool(readEnv("VITE_USE_MOCK", "REACT_APP_USE_MOCK"));
const apiKey = readEnv("VITE_API_KEY", "REACT_APP_API_KEY");
const disableBackendVideo = asBool(
  readEnv("VITE_DISABLE_BACKEND_VIDEO", "REACT_APP_DISABLE_BACKEND_VIDEO")
);
const disableJanus = asBool(
  readEnv("VITE_DISABLE_JANUS", "REACT_APP_DISABLE_JANUS")
);
const metadataLatest =
  readEnv("VITE_METADATA_LATEST_URL", "REACT_APP_METADATA_LATEST_URL") ??
  `${normalizedRestBase}/api/realtime/latest`;
const metadataPollMs = asPositiveInt(
  readEnv("VITE_METADATA_POLL_MS", "REACT_APP_METADATA_POLL_MS"),
  250
);
const videoPrimaryTransport = normalizePrimaryTransport(
  readEnv("VITE_VIDEO_PRIMARY_TRANSPORT", "REACT_APP_VIDEO_PRIMARY_TRANSPORT"),
  "janus"
);
const videoFallbackTransport = normalizeFallbackTransport(
  readEnv("VITE_VIDEO_FALLBACK_TRANSPORT", "REACT_APP_VIDEO_FALLBACK_TRANSPORT"),
  "wsjpeg"
);
const videoTransportPlan = buildTransportPlan(videoPrimaryTransport, videoFallbackTransport);

export const API = {
  REST_BASE: normalizedRestBase,
  WS_METADATA: wsMetadata,
  JANUS_HTTP: janusHttp,
  JANUS_MOUNTPOINT: Number.isFinite(janusMountpoint) && janusMountpoint > 0 ? janusMountpoint : 1,
  DISABLE_JANUS: disableJanus,
  DISABLE_BACKEND_VIDEO: disableBackendVideo,
  VIDEO_PRIMARY_TRANSPORT: videoPrimaryTransport,
  VIDEO_FALLBACK_TRANSPORT: videoFallbackTransport,
  VIDEO_TRANSPORT_PLAN: videoTransportPlan,
  METADATA_LATEST: metadataLatest,
  METADATA_POLL_MS: metadataPollMs,
  USE_MOCK: useMock || false,
  API_KEY: apiKey
};
