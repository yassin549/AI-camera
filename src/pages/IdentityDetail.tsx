import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import {
  deleteIdentity,
  getIdentityById,
  mergeIdentity,
  renameIdentity,
  type IdentityDetailData,
  type IdentitySummary
} from "../api/client";

interface DetailRouteState {
  layoutId?: string;
  identity?: IdentitySummary;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function IdentityDetail(): JSX.Element {
  const navigate = useNavigate();
  const params = useParams();
  const location = useLocation();
  const routeState = (location.state ?? {}) as DetailRouteState;
  const identityId = params.id ?? "";
  const hasRouteIdentity = routeState.identity?.id === identityId;

  const [detail, setDetail] = useState<IdentityDetailData | null>(
    routeState.identity
      ? {
          ...routeState.identity,
          timeline: [
            { at: routeState.identity.first_seen, note: "First observed" },
            { at: routeState.identity.last_seen, note: "Most recent observation" }
          ],
          samples: [...routeState.identity.face_samples, ...routeState.identity.body_samples]
        }
      : null
  );
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(!detail);

  useEffect(() => {
    if (hasRouteIdentity) {
      return;
    }
    let cancelled = false;
    const load = async () => {
      if (!identityId) {
        return;
      }
      setLoading(true);
      try {
        const payload = await getIdentityById(identityId);
        if (!cancelled) {
          setDetail(payload);
        }
      } catch (err) {
        if (!cancelled) {
          setStatus(err instanceof Error ? err.message : "Unable to load identity detail");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [hasRouteIdentity, identityId]);

  const samples = useMemo(() => {
    if (!detail) {
      return [];
    }
    if (detail.samples?.length) {
      return detail.samples;
    }
    return [...detail.face_samples, ...detail.body_samples];
  }, [detail]);

  const handleRename = async () => {
    if (!detail) {
      return;
    }
    const nextName = window.prompt("Rename identity to:", detail.id);
    if (!nextName || nextName === detail.id) {
      return;
    }
    try {
      setStatus("Renaming...");
      await renameIdentity(detail.id, nextName);
      setStatus("Rename complete");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Rename failed");
    }
  };

  const handleMerge = async () => {
    if (!detail) {
      return;
    }
    const targetId = window.prompt("Merge into identity:", "");
    if (!targetId) {
      return;
    }
    try {
      setStatus("Merging...");
      await mergeIdentity(detail.id, targetId);
      setStatus(`Merged into ${targetId}`);
      navigate(`/library/${targetId}`, { replace: true });
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Merge failed");
    }
  };

  const handleDelete = async () => {
    if (!detail) {
      return;
    }
    const confirmed = window.confirm(`Delete identity ${detail.id}?`);
    if (!confirmed) {
      return;
    }
    try {
      setStatus("Deleting...");
      await deleteIdentity(detail.id);
      navigate("/library");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Delete failed");
    }
  };

  const layoutId = routeState.layoutId ?? `identity-card-${identityId}`;

  return (
    <section className="h-full overflow-auto px-4 pb-4 pt-2">
      <motion.article
        key={identityId}
        layoutId={layoutId}
        className="glass-panel min-h-[540px] rounded-2xl p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.12, ease: "easeOut" }}
      >
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="rounded-lg bg-white/10 px-3 py-1.5 text-sm text-slate-100 hover:bg-white/20"
        >
          Back
        </button>

        {loading ? (
          <div className="mt-6 text-sm text-slate-300">Loading detail...</div>
        ) : (
          <>
            <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
              <h1 className="text-2xl font-semibold text-slate-100">Identity {detail?.id}</h1>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  className="rounded-lg bg-accent px-3 py-1.5 text-sm text-white hover:bg-accentSoft"
                  onClick={handleRename}
                >
                  Rename
                </button>
                <button
                  type="button"
                  className="rounded-lg bg-white/10 px-3 py-1.5 text-sm text-slate-100 hover:bg-white/20"
                  onClick={handleMerge}
                >
                  Merge
                </button>
                <button
                  type="button"
                  className="rounded-lg bg-rose-500/80 px-3 py-1.5 text-sm text-white hover:bg-rose-400"
                  onClick={handleDelete}
                >
                  Delete
                </button>
              </div>
            </div>

            <div className="mt-3 grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-white/10 bg-slate-950/30 p-4">
                <div className="text-xs uppercase tracking-[0.14em] text-slate-400">Summary</div>
                <div className="mt-3 text-sm text-slate-200">First seen: {formatDate(detail?.first_seen ?? "")}</div>
                <div className="mt-2 text-sm text-slate-200">Last seen: {formatDate(detail?.last_seen ?? "")}</div>
                <div className="mt-2 text-sm text-slate-200">
                  Frequency: {detail?.stats?.frequency ?? detail?.stats?.sightings ?? 0}
                </div>
              </div>
              <div className="rounded-xl border border-white/10 bg-slate-950/30 p-4">
                <div className="text-xs uppercase tracking-[0.14em] text-slate-400">Thumbnails</div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  {(detail?.face_samples ?? []).slice(0, 2).map((sample) => (
                    <img key={sample} src={sample} alt="Face sample" className="h-24 w-full rounded-lg object-cover" />
                  ))}
                  {(detail?.body_samples ?? []).slice(0, 2).map((sample) => (
                    <img key={sample} src={sample} alt="Body sample" className="h-24 w-full rounded-lg object-cover" />
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-white/10 bg-slate-950/30 p-4">
              <div className="text-xs uppercase tracking-[0.14em] text-slate-400">Timeline</div>
              <div className="mt-3 space-y-2">
                {(detail?.timeline ?? []).map((item) => (
                  <div key={`${item.at}-${item.note}`} className="rounded-lg bg-white/5 px-3 py-2 text-sm text-slate-200">
                    <span className="font-medium text-slate-100">{formatDate(item.at)}</span>
                    <span className="ml-3 text-slate-300">{item.note}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-white/10 bg-slate-950/30 p-4">
              <div className="text-xs uppercase tracking-[0.14em] text-slate-400">Sample Carousel</div>
              <div className="mt-3 flex gap-3 overflow-x-auto pb-2">
                {samples.map((sample) => (
                  <img
                    key={sample}
                    src={sample}
                    alt="Identity sample"
                    className="h-28 w-44 flex-none rounded-lg object-cover"
                    loading="lazy"
                  />
                ))}
              </div>
            </div>
          </>
        )}

        {status ? <div className="mt-4 text-sm text-accentSoft">{status}</div> : null}
      </motion.article>
    </section>
  );
}
