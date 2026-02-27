/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  readonly VITE_API_KEY?: string;
  readonly VITE_WS_METADATA_URL?: string;
  readonly VITE_WEBRTC_OFFER?: string;
  readonly VITE_JANUS_HTTP_URL?: string;
  readonly VITE_JANUS_MOUNTPOINT?: string;
  readonly VITE_DIRECT_STREAM_URL?: string;
  readonly VITE_DIRECT_STREAM_KIND?: string;
  readonly VITE_DISABLE_BACKEND_VIDEO?: string;
  readonly VITE_DISABLE_JANUS?: string;
  readonly VITE_METADATA_LATEST_URL?: string;
  readonly VITE_METADATA_POLL_MS?: string;
  readonly VITE_USE_MOCK?: string;
  readonly VITE_DISABLE_WEBRTC?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare const __APP_ENV__: Record<string, string | undefined>;
