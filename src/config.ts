type EnvMap = Record<string, string | undefined>;

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

const defaultRestBase = "http://localhost:8080";
const restBase = readEnv("VITE_API_BASE", "REACT_APP_API_URL") ?? defaultRestBase;
const normalizedRestBase = trimTrailingSlash(restBase || defaultRestBase);
const wsMetadata =
  readEnv("VITE_WS_METADATA_URL", "REACT_APP_WS_URL") ??
  `${normalizedRestBase}/api/realtime/ws`;
const webrtcOffer = readEnv("VITE_WEBRTC_OFFER", "REACT_APP_WEBRTC_OFFER");
const janusHttp = readEnv("VITE_JANUS_HTTP_URL", "REACT_APP_JANUS_HTTP_URL") ?? "http://localhost:8088/janus";
const janusMountpoint = Number(readEnv("VITE_JANUS_MOUNTPOINT", "REACT_APP_JANUS_MOUNTPOINT") ?? "1");
const useMock = asBool(readEnv("VITE_USE_MOCK", "REACT_APP_USE_MOCK"));
const apiKey = readEnv("VITE_API_KEY", "REACT_APP_API_KEY");
const directStreamUrl = readEnv("VITE_DIRECT_STREAM_URL", "REACT_APP_DIRECT_STREAM_URL");
const directStreamKind = (
  readEnv("VITE_DIRECT_STREAM_KIND", "REACT_APP_DIRECT_STREAM_KIND") ?? "auto"
).trim().toLowerCase();
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

export const API = {
  REST_BASE: normalizedRestBase,
  WS_METADATA: wsMetadata,
  WEBRTC_OFFER: webrtcOffer ?? `${normalizedRestBase}/webrtc/offer`,
  JANUS_HTTP: janusHttp,
  JANUS_MOUNTPOINT: Number.isFinite(janusMountpoint) && janusMountpoint > 0 ? janusMountpoint : 1,
  DIRECT_STREAM_URL: directStreamUrl ?? "",
  DIRECT_STREAM_KIND:
    directStreamKind === "image" || directStreamKind === "video" ? directStreamKind : "auto",
  DISABLE_JANUS: disableJanus,
  DISABLE_BACKEND_VIDEO: disableBackendVideo,
  METADATA_LATEST: metadataLatest,
  METADATA_POLL_MS: metadataPollMs,
  USE_MOCK: useMock || false,
  API_KEY: apiKey
};
