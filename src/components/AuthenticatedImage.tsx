import { useEffect, useMemo, useState } from "react";
import { withApiKeyHeaders } from "../api/client";

interface AuthenticatedImageProps {
  src: string;
  alt: string;
  className?: string;
  loading?: "eager" | "lazy";
}

const objectUrlCache = new Map<string, string>();

function isNgrokFreeUrl(input: string): boolean {
  try {
    const host = new URL(input).hostname.toLowerCase();
    return host.endsWith(".ngrok-free.app") || host.endsWith(".ngrok-free.dev");
  } catch {
    return false;
  }
}

export function AuthenticatedImage({
  src,
  alt,
  className,
  loading = "lazy"
}: AuthenticatedImageProps): JSX.Element {
  const [resolvedSrc, setResolvedSrc] = useState<string>(src);
  const shouldProxyFetch = useMemo(() => Boolean(src) && isNgrokFreeUrl(src), [src]);

  useEffect(() => {
    let cancelled = false;
    if (!src) {
      setResolvedSrc("");
      return;
    }
    if (!shouldProxyFetch) {
      setResolvedSrc(src);
      return;
    }
    const cached = objectUrlCache.get(src);
    if (cached) {
      setResolvedSrc(cached);
      return;
    }

    const controller = new AbortController();
    void fetch(src, {
      method: "GET",
      headers: withApiKeyHeaders(),
      signal: controller.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Image fetch failed (${response.status})`);
        }
        const blob = await response.blob();
        if (cancelled) {
          return;
        }
        const objectUrl = URL.createObjectURL(blob);
        objectUrlCache.set(src, objectUrl);
        setResolvedSrc(objectUrl);
      })
      .catch(() => {
        if (!cancelled) {
          setResolvedSrc(src);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [shouldProxyFetch, src]);

  return <img src={resolvedSrc} alt={alt} loading={loading} className={className} />;
}

