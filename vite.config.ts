import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const useMock =
    process.argv.includes("--mock") ||
    mode === "mock" ||
    env.VITE_USE_MOCK === "1" ||
    process.env.VITE_USE_MOCK === "1";
  const httpTarget = useMock
    ? "http://localhost:8787"
    : env.VITE_PROXY_TARGET || process.env.VITE_PROXY_TARGET || "http://localhost:8080";
  const wsTarget = httpTarget.replace(/^http/i, "ws");
  const appEnv = {
    VITE_API_BASE: env.VITE_API_BASE ?? process.env.VITE_API_BASE,
    VITE_API_KEY: env.VITE_API_KEY ?? process.env.VITE_API_KEY,
    VITE_WS_METADATA_URL: env.VITE_WS_METADATA_URL ?? process.env.VITE_WS_METADATA_URL,
    VITE_WEBRTC_OFFER: env.VITE_WEBRTC_OFFER ?? process.env.VITE_WEBRTC_OFFER,
    VITE_JANUS_HTTP_URL: env.VITE_JANUS_HTTP_URL ?? process.env.VITE_JANUS_HTTP_URL,
    VITE_JANUS_MOUNTPOINT: env.VITE_JANUS_MOUNTPOINT ?? process.env.VITE_JANUS_MOUNTPOINT,
    VITE_DIRECT_STREAM_URL: env.VITE_DIRECT_STREAM_URL ?? process.env.VITE_DIRECT_STREAM_URL,
    VITE_DIRECT_STREAM_KIND: env.VITE_DIRECT_STREAM_KIND ?? process.env.VITE_DIRECT_STREAM_KIND,
    VITE_DISABLE_BACKEND_VIDEO:
      env.VITE_DISABLE_BACKEND_VIDEO ?? process.env.VITE_DISABLE_BACKEND_VIDEO,
    VITE_DISABLE_JANUS: env.VITE_DISABLE_JANUS ?? process.env.VITE_DISABLE_JANUS,
    VITE_METADATA_LATEST_URL: env.VITE_METADATA_LATEST_URL ?? process.env.VITE_METADATA_LATEST_URL,
    VITE_METADATA_POLL_MS: env.VITE_METADATA_POLL_MS ?? process.env.VITE_METADATA_POLL_MS,
    VITE_USE_MOCK: env.VITE_USE_MOCK ?? process.env.VITE_USE_MOCK,
    VITE_DISABLE_WEBRTC: env.VITE_DISABLE_WEBRTC ?? process.env.VITE_DISABLE_WEBRTC,
    REACT_APP_API_URL: process.env.REACT_APP_API_URL,
    REACT_APP_API_KEY: process.env.REACT_APP_API_KEY,
    REACT_APP_WS_URL: process.env.REACT_APP_WS_URL,
    REACT_APP_WEBRTC_OFFER: process.env.REACT_APP_WEBRTC_OFFER,
    REACT_APP_JANUS_HTTP_URL: process.env.REACT_APP_JANUS_HTTP_URL,
    REACT_APP_JANUS_MOUNTPOINT: process.env.REACT_APP_JANUS_MOUNTPOINT,
    REACT_APP_DIRECT_STREAM_URL: process.env.REACT_APP_DIRECT_STREAM_URL,
    REACT_APP_DIRECT_STREAM_KIND: process.env.REACT_APP_DIRECT_STREAM_KIND,
    REACT_APP_DISABLE_BACKEND_VIDEO: process.env.REACT_APP_DISABLE_BACKEND_VIDEO,
    REACT_APP_METADATA_LATEST_URL: process.env.REACT_APP_METADATA_LATEST_URL,
    REACT_APP_METADATA_POLL_MS: process.env.REACT_APP_METADATA_POLL_MS,
    REACT_APP_USE_MOCK: process.env.REACT_APP_USE_MOCK,
    NODE_ENV: process.env.NODE_ENV
  };

  return {
    plugins: [react()],
    define: {
      __APP_ENV__: JSON.stringify(appEnv)
    },
    server: {
      port: 5173,
      proxy: {
        "/api": { target: httpTarget, changeOrigin: true },
        "/webrtc": { target: httpTarget, changeOrigin: true },
        "/media": { target: httpTarget, changeOrigin: true },
        "/faces": { target: httpTarget, changeOrigin: true },
        "/ws": { target: wsTarget, ws: true, changeOrigin: true }
      }
    }
  };
});
