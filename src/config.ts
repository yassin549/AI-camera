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

function asBoolWithDefault(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) {
    return fallback;
  }
  return asBool(value);
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

function asNonNegativeInt(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 0) {
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
  readEnv(
    "VITE_JANUS_HTTP_URL",
    "VITE_JANUS_URL",
    "REACT_APP_JANUS_HTTP_URL",
    "REACT_APP_JANUS_URL"
  ) ??
  `${normalizedRestBase}/janus`;
const janusWsRaw = readEnv("VITE_JANUS_WS_URL", "REACT_APP_JANUS_WS_URL") ?? "";
const janusWs = janusWsRaw ? trimTrailingSlash(janusWsRaw) : "";
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
const janusFirstTrackTimeoutMs = asPositiveInt(
  readEnv("VITE_JANUS_FIRST_TRACK_TIMEOUT_MS", "REACT_APP_JANUS_FIRST_TRACK_TIMEOUT_MS"),
  15000
);
const wsJpegFpsLocal = asPositiveInt(
  readEnv("VITE_WSJPEG_FPS_LOCAL", "REACT_APP_WSJPEG_FPS_LOCAL"),
  20
);
const wsJpegFpsRemote = asPositiveInt(
  readEnv("VITE_WSJPEG_FPS_REMOTE", "REACT_APP_WSJPEG_FPS_REMOTE"),
  10
);
const wsJpegAdaptive = asBoolWithDefault(
  readEnv("VITE_WSJPEG_ADAPTIVE", "REACT_APP_WSJPEG_ADAPTIVE"),
  true
);
const wsJpegRenderMaxFps = asNonNegativeInt(
  readEnv("VITE_WSJPEG_RENDER_MAX_FPS", "REACT_APP_WSJPEG_RENDER_MAX_FPS"),
  0
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
  JANUS_WS: janusWs,
  JANUS_MOUNTPOINT: Number.isFinite(janusMountpoint) && janusMountpoint > 0 ? janusMountpoint : 1,
  DISABLE_JANUS: disableJanus,
  DISABLE_BACKEND_VIDEO: disableBackendVideo,
  VIDEO_PRIMARY_TRANSPORT: videoPrimaryTransport,
  VIDEO_FALLBACK_TRANSPORT: videoFallbackTransport,
  VIDEO_TRANSPORT_PLAN: videoTransportPlan,
  METADATA_LATEST: metadataLatest,
  METADATA_POLL_MS: metadataPollMs,
  JANUS_FIRST_TRACK_TIMEOUT_MS: janusFirstTrackTimeoutMs,
  WSJPEG_FPS_LOCAL: wsJpegFpsLocal,
  WSJPEG_FPS_REMOTE: wsJpegFpsRemote,
  WSJPEG_ADAPTIVE: wsJpegAdaptive,
  WSJPEG_RENDER_MAX_FPS: wsJpegRenderMaxFps,
  USE_MOCK: useMock || false,
  API_KEY: apiKey
};
