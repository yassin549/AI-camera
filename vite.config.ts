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
    VITE_WS_METADATA_URL: env.VITE_WS_METADATA_URL ?? process.env.VITE_WS_METADATA_URL,
    VITE_WEBRTC_OFFER: env.VITE_WEBRTC_OFFER ?? process.env.VITE_WEBRTC_OFFER,
    VITE_JANUS_HTTP_URL: env.VITE_JANUS_HTTP_URL ?? process.env.VITE_JANUS_HTTP_URL,
    VITE_JANUS_MOUNTPOINT: env.VITE_JANUS_MOUNTPOINT ?? process.env.VITE_JANUS_MOUNTPOINT,
    VITE_VIDEO_WS_URL: env.VITE_VIDEO_WS_URL ?? process.env.VITE_VIDEO_WS_URL,
    VITE_MJPEG: env.VITE_MJPEG ?? process.env.VITE_MJPEG,
    VITE_USE_MOCK: env.VITE_USE_MOCK ?? process.env.VITE_USE_MOCK,
    VITE_DISABLE_WEBRTC: env.VITE_DISABLE_WEBRTC ?? process.env.VITE_DISABLE_WEBRTC,
    REACT_APP_API_URL: process.env.REACT_APP_API_URL,
    REACT_APP_WS_URL: process.env.REACT_APP_WS_URL,
    REACT_APP_WEBRTC_OFFER: process.env.REACT_APP_WEBRTC_OFFER,
    REACT_APP_JANUS_HTTP_URL: process.env.REACT_APP_JANUS_HTTP_URL,
    REACT_APP_JANUS_MOUNTPOINT: process.env.REACT_APP_JANUS_MOUNTPOINT,
    REACT_APP_VIDEO_WS_URL: process.env.REACT_APP_VIDEO_WS_URL,
    REACT_APP_MJPEG: process.env.REACT_APP_MJPEG,
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
        "/stream.mjpeg": { target: httpTarget, changeOrigin: true },
        "/media": { target: httpTarget, changeOrigin: true },
        "/faces": { target: httpTarget, changeOrigin: true },
        "/ws": { target: wsTarget, ws: true, changeOrigin: true }
      }
    }
  };
});
