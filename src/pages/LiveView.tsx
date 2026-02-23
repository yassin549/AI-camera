import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { muteIdentity, snapshotIdentity, type TrackPayload } from "../api/client";
import { VideoCanvas } from "../components/VideoCanvas";

interface LiveViewProps {
  active: boolean;
}

export function LiveView({ active }: LiveViewProps): JSX.Element {
  const navigate = useNavigate();
  const [pixelPerfect, setPixelPerfect] = useState(true);
  const [focusedTrack, setFocusedTrack] = useState<TrackPayload | null>(null);
  const [roster, setRoster] = useState<TrackPayload[]>([]);
  const [actionState, setActionState] = useState<string>("");

  const uniqueRoster = useMemo(() => {
    const map = new Map<string, TrackPayload>();
    for (const track of roster) {
      const key = track.identity_id !== null ? `id-${track.identity_id}` : `track-${track.track_id}`;
      if (!map.has(key)) {
        map.set(key, track);
      }
    }
    return [...map.values()];
  }, [roster]);

  useEffect(() => {
    if (focusedTrack && !roster.some((track) => track.track_id === focusedTrack.track_id)) {
      setFocusedTrack(null);
    }
  }, [focusedTrack, roster]);

  const runQuickAction = async (action: "snapshot" | "mute") => {
    if (focusedTrack?.identity_id === null || focusedTrack?.identity_id === undefined) {
      return;
    }
    const id = String(focusedTrack.identity_id);
    try {
      setActionState(`${action}...`);
      if (action === "snapshot") {
        await snapshotIdentity(id);
      } else {
        await muteIdentity(id);
      }
      setActionState(`${action} done`);
      window.setTimeout(() => setActionState(""), 1200);
    } catch {
      setActionState(`${action} failed`);
      window.setTimeout(() => setActionState(""), 1800);
    }
  };

  return (
    <div
      className={`absolute inset-0 grid grid-cols-1 gap-4 px-4 pb-4 pt-2 xl:grid-cols-[minmax(0,1fr)_320px] ${
        active ? "opacity-100" : "pointer-events-none opacity-0"
      } transition-opacity duration-200`}
      aria-hidden={!active}
    >
      <motion.section
        className="glass-panel relative min-h-[540px] overflow-hidden rounded-2xl p-3"
        initial={false}
        animate={{ scale: active ? 1 : 0.99, opacity: active ? 1 : 0.4 }}
        transition={{ duration: 0.2 }}
      >
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-slate-200">Live View</div>
            <div className="text-xs text-slate-400">Raw stream stays untouched. Overlays are rendered client-side.</div>
          </div>
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-white/10 px-3 py-1.5 text-xs text-slate-200">
            <input
              type="checkbox"
              className="accent-accent"
              checked={pixelPerfect}
              onChange={(event) => setPixelPerfect(event.target.checked)}
              aria-label="Enable pixel perfect overlay rendering"
            />
            Pixel-perfect
          </label>
        </div>
        <div className="h-[calc(100%-3rem)]">
          <VideoCanvas
            active={active}
            pixelPerfect={pixelPerfect}
            focusedTrackId={focusedTrack?.track_id ?? null}
            onTrackFocus={setFocusedTrack}
            onRosterChange={setRoster}
          />
        </div>
      </motion.section>

      <aside className="glass-panel flex min-h-[540px] flex-col rounded-2xl p-4">
        <h2 className="text-sm font-semibold text-slate-200">Realtime roster</h2>
        <p className="mt-1 text-xs text-slate-400">Observed identities in the current scene.</p>
        <div className="mt-4 space-y-2 overflow-auto">
          {uniqueRoster.length === 0 ? (
            <div className="rounded-xl border border-white/10 bg-slate-950/30 p-3 text-xs text-slate-400">
              Waiting for metadata...
            </div>
          ) : (
            uniqueRoster.map((track) => {
              const focused = focusedTrack?.track_id === track.track_id;
              return (
                <button
                  key={track.track_id}
                  type="button"
                  className={`w-full rounded-xl border px-3 py-2 text-left text-xs transition ${
                    focused
                      ? "border-amber-300/70 bg-amber-200/10 text-amber-100"
                      : "border-white/10 bg-slate-950/30 text-slate-200 hover:border-accent/60 hover:bg-accent/10"
                  }`}
                  onClick={() => setFocusedTrack(track)}
                >
                  <div className="font-medium">{track.label}</div>
                  <div className="mt-1 text-slate-400">Track #{track.track_id}</div>
                </button>
              );
            })
          )}
        </div>

        <div className="mt-4 rounded-xl border border-white/10 bg-slate-950/40 p-3">
          <div className="text-xs font-medium text-slate-200">Focused identity</div>
          {focusedTrack?.identity_id !== null && focusedTrack?.identity_id !== undefined ? (
            <>
              <div className="mt-2 text-sm text-slate-100">ID {focusedTrack.identity_id}</div>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accentSoft"
                  onClick={() => navigate(`/library/${focusedTrack.identity_id}`)}
                >
                  Go to library
                </button>
                <button
                  type="button"
                  className="rounded-lg bg-white/10 px-3 py-1.5 text-xs text-slate-100 hover:bg-white/20"
                  onClick={() => runQuickAction("snapshot")}
                >
                  Snapshot
                </button>
                <button
                  type="button"
                  className="rounded-lg bg-white/10 px-3 py-1.5 text-xs text-slate-100 hover:bg-white/20"
                  onClick={() => runQuickAction("mute")}
                >
                  Mute
                </button>
              </div>
            </>
          ) : (
            <div className="mt-2 text-xs text-slate-400">Click a box or roster row to focus.</div>
          )}
          {actionState ? <div className="mt-2 text-xs text-accentSoft">{actionState}</div> : null}
        </div>
      </aside>
    </div>
  );
}
