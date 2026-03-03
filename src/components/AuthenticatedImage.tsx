import { useEffect, useMemo, useState } from "react";
import { withApiKeyHeaders, withApiKeyQuery } from "../api/client";
import { API } from "../config";

interface AuthenticatedImageProps {
  src: string;
  alt: string;
  className?: string;
  loading?: "eager" | "lazy";
}

const objectUrlCache = new Map<string, string>();

function isNgrokFreeUrl(input: string): boolean {
  try {
    const host = new URL(input, window.location.href).hostname.toLowerCase();
    return host.endsWith(".ngrok-free.app") || host.endsWith(".ngrok-free.dev");
  } catch {
    return false;
  }
}

function normalizeMediaUrl(src: string): string {
  if (!src) {
    return src;
  }
  if (/^(https?:)?\/\//i.test(src) || src.startsWith("data:") || src.startsWith("blob:")) {
    return src;
  }
  const base = (API.REST_BASE || "").replace(/\/+$/, "");
  if (!base) {
    return src;
  }
  const path = src.startsWith("/") ? src : `/${src}`;
  return `${base}${path}`;
}

export function AuthenticatedImage({
  src,
  alt,
  className,
  loading = "lazy"
}: AuthenticatedImageProps): JSX.Element {
  const normalizedSrc = useMemo(() => withApiKeyQuery(normalizeMediaUrl(src)), [src]);
  const [resolvedSrc, setResolvedSrc] = useState<string>(normalizedSrc);
  const shouldProxyFetch = useMemo(
    () => Boolean(normalizedSrc) && isNgrokFreeUrl(normalizedSrc),
    [normalizedSrc]
  );

  useEffect(() => {
    let cancelled = false;
    if (!normalizedSrc) {
      setResolvedSrc("");
      return;
    }
    if (!shouldProxyFetch) {
      setResolvedSrc(normalizedSrc);
      return;
    }
    const cached = objectUrlCache.get(normalizedSrc);
    if (cached) {
      setResolvedSrc(cached);
      return;
    }

    const controller = new AbortController();
    void fetch(normalizedSrc, {
      method: "GET",
      headers: withApiKeyHeaders(),
      signal: controller.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Image fetch failed (${response.status})`);
        }
        const contentType = (response.headers.get("content-type") || "").toLowerCase();
        if (!contentType.startsWith("image/")) {
          throw new Error(`Unexpected image content-type: ${contentType || "unknown"}`);
        }
        const blob = await response.blob();
        if (cancelled) {
          return;
        }
        const objectUrl = URL.createObjectURL(blob);
        objectUrlCache.set(normalizedSrc, objectUrl);
        setResolvedSrc(objectUrl);
      })
      .catch(() => {
        if (!cancelled) {
          setResolvedSrc(normalizedSrc);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [normalizedSrc, shouldProxyFetch]);

  return <img src={resolvedSrc} alt={alt} loading={loading} className={className} />;
}
