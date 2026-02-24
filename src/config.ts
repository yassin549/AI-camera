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

const restBase = readEnv("VITE_API_BASE", "REACT_APP_API_URL") ?? "http://localhost:8080";
const wsMetadata = readEnv("VITE_WS_METADATA_URL", "REACT_APP_WS_URL") ?? "ws://localhost:8080/ws/metadata";
const webrtcOffer = readEnv("VITE_WEBRTC_OFFER", "REACT_APP_WEBRTC_OFFER");
const janusHttp = readEnv("VITE_JANUS_HTTP_URL", "REACT_APP_JANUS_HTTP_URL") ?? "http://localhost:8088/janus";
const janusMountpoint = Number(readEnv("VITE_JANUS_MOUNTPOINT", "REACT_APP_JANUS_MOUNTPOINT") ?? "1");
const videoWs = readEnv("VITE_VIDEO_WS_URL", "REACT_APP_VIDEO_WS_URL") ?? "/api/media/ws";
const mjpegFallback = readEnv("VITE_MJPEG", "REACT_APP_MJPEG") ?? "/stream.mjpeg";
const useMock = asBool(readEnv("VITE_USE_MOCK", "REACT_APP_USE_MOCK"));

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export const API = {
  REST_BASE: trimTrailingSlash(restBase),
  WS_METADATA: wsMetadata,
  WEBRTC_OFFER:
    webrtcOffer ?? `${trimTrailingSlash(restBase || "http://localhost:8080")}/webrtc/offer`,
  JANUS_HTTP: janusHttp,
  JANUS_MOUNTPOINT: Number.isFinite(janusMountpoint) && janusMountpoint > 0 ? janusMountpoint : 1,
  VIDEO_WS: videoWs,
  MJPEG_FALLBACK: mjpegFallback,
  USE_MOCK: useMock || false
};
