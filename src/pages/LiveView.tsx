import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { assignTrackIdentity, getIdentities, muteIdentity, snapshotIdentity, type TrackPayload } from "../api/client";
import { VideoCanvas } from "../components/VideoCanvas";

interface LiveViewProps {
  active: boolean;
}

export function LiveView({ active }: LiveViewProps): JSX.Element {
  const navigate = useNavigate();
  const [pixelPerfect, setPixelPerfect] = useState(false);
  const [focusedTrack, setFocusedTrack] = useState<TrackPayload | null>(null);
  const [roster, setRoster] = useState<TrackPayload[]>([]);
  const [identityOptions, setIdentityOptions] = useState<{ id: string; label: string }[]>([]);
  const [linkTargetId, setLinkTargetId] = useState<string>("");
  const [actionState, setActionState] = useState<string>("");

  const uniqueRoster = useMemo(() => {
    const map = new Map<string, TrackPayload>();
    for (const track of roster) {
      const key = track.identity_id !== null ? `id-${track.identity_id}` : `track-${track.track_id}`;
      const current = map.get(key);
      const currentAge = current?.age_ratio ?? 1;
      const nextAge = track.age_ratio ?? 1;
      if (!current || nextAge <= currentAge) {
        map.set(key, track);
      }
    }
    return [...map.values()];
  }, [roster]);

  const focusedIdentityId = focusedTrack?.identity_id ?? null;
  const hasFocusedTrack = focusedTrack !== null;
  const canRunIdentityActions = focusedIdentityId !== null;
  const actionDisabledReason = !hasFocusedTrack
    ? "Select a track first."
    : !canRunIdentityActions
      ? "Track is unresolved. Link it to an identity first."
      : "";

  useEffect(() => {
    if (!active) {
      setIdentityOptions([]);
      setLinkTargetId("");
      return;
    }
    let cancelled = false;

    const loadIdentityOptions = async () => {
      try {
        const identities = await getIdentities();
        if (cancelled) {
          return;
        }
        const next = identities.map((identity) => ({
          id: String(identity.id),
          label: identity.display_name || identity.name || `Identity ${identity.id}`
        }));
        setIdentityOptions(next);
      } catch {
        if (!cancelled) {
          setIdentityOptions([]);
        }
      }
    };

    void loadIdentityOptions();
    const timer = window.setInterval(() => {
      void loadIdentityOptions();
    }, 12000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [active]);

  useEffect(() => {
    if (!focusedTrack) {
      return;
    }
    const latest = roster.find((track) => track.track_id === focusedTrack.track_id);
    if (!latest) {
      setFocusedTrack(null);
      return;
    }
    const sameIdentity = latest.identity_id === focusedTrack.identity_id;
    const sameLabel = latest.label === focusedTrack.label;
    const sameMuted = Boolean(latest.muted) === Boolean(focusedTrack.muted);
    if (!sameIdentity || !sameLabel || !sameMuted) {
      setFocusedTrack(latest);
    }
  }, [focusedTrack, roster]);

  useEffect(() => {
    if (!focusedTrack || focusedTrack.identity_id !== null) {
      return;
    }
    if (!identityOptions.length) {
      if (linkTargetId !== "") {
        setLinkTargetId("");
      }
      return;
    }
    const hasCurrent = identityOptions.some((item) => item.id === linkTargetId);
    if (!hasCurrent) {
      setLinkTargetId(identityOptions[0].id);
    }
  }, [focusedTrack, identityOptions, linkTargetId]);

  const runQuickAction = async (action: "snapshot" | "mute") => {
    if (focusedTrack?.identity_id === null || focusedTrack?.identity_id === undefined) {
      setActionState("Action blocked: track has no linked identity");
      window.setTimeout(() => setActionState(""), 1600);
      return;
    }
    const id = String(focusedTrack.identity_id);
    try {
      setActionState(`${action}...`);
      if (action === "snapshot") {
        await snapshotIdentity(id);
      } else {
        const nextMuted = !Boolean(focusedTrack.muted);
        const response = await muteIdentity(id, nextMuted);
        setRoster((prev) =>
          prev.map((track) =>
            track.identity_id === focusedTrack.identity_id
              ? { ...track, muted: response.muted }
              : track
          )
        );
        setFocusedTrack((prev) => (prev ? { ...prev, muted: response.muted } : prev));
      }
      setActionState(`${action} done`);
      window.setTimeout(() => setActionState(""), 1200);
    } catch {
      setActionState(`${action} failed`);
      window.setTimeout(() => setActionState(""), 1800);
    }
  };

  const runManualLink = async () => {
    if (!focusedTrack) {
      setActionState("No track selected");
      window.setTimeout(() => setActionState(""), 1400);
      return;
    }
    if (focusedTrack.identity_id !== null) {
      setActionState("Track is already linked");
      window.setTimeout(() => setActionState(""), 1400);
      return;
    }
    if (!linkTargetId) {
      setActionState("Select an identity to link");
      window.setTimeout(() => setActionState(""), 1600);
      return;
    }
    try {
      setActionState("linking...");
      const payload = await assignTrackIdentity(focusedTrack.track_id, linkTargetId);
      const assignedIdentityId = Number(payload.identity_id);
      const linkedLabel = `ID:${assignedIdentityId} (1.00)`;
      setRoster((prev) =>
        prev.map((track) =>
          track.track_id === focusedTrack.track_id
            ? {
                ...track,
                identity_id: assignedIdentityId,
                modality: "manual",
                label: linkedLabel
              }
            : track
        )
      );
      setFocusedTrack((prev) =>
        prev && prev.track_id === focusedTrack.track_id
          ? {
              ...prev,
              identity_id: assignedIdentityId,
              modality: "manual",
              label: linkedLabel
            }
          : prev
      );
      setActionState("link done");
      window.setTimeout(() => setActionState(""), 1300);
    } catch {
      setActionState("link failed");
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
          {focusedTrack ? (
            <div className="mt-2 text-sm text-slate-100">
              {focusedTrack.identity_id !== null ? `ID ${focusedTrack.identity_id}` : `Track #${focusedTrack.track_id} unresolved`}
            </div>
          ) : (
            <div className="mt-2 text-xs text-slate-400">Click a box or roster row to focus.</div>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accentSoft disabled:cursor-not-allowed disabled:opacity-45"
              disabled={!canRunIdentityActions}
              onClick={() => {
                if (focusedTrack?.identity_id !== null && focusedTrack?.identity_id !== undefined) {
                  navigate(`/library/${focusedTrack.identity_id}`);
                }
              }}
            >
              Go to library
            </button>
            <button
              type="button"
              className="rounded-lg bg-white/10 px-3 py-1.5 text-xs text-slate-100 hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-45"
              disabled={!canRunIdentityActions}
              onClick={() => runQuickAction("snapshot")}
            >
              Snapshot
            </button>
            <button
              type="button"
              className="rounded-lg bg-white/10 px-3 py-1.5 text-xs text-slate-100 hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-45"
              disabled={!canRunIdentityActions}
              onClick={() => runQuickAction("mute")}
            >
              {focusedTrack?.muted ? "Unmute" : "Mute"}
            </button>
          </div>
          {!canRunIdentityActions && actionDisabledReason ? (
            <div className="mt-2 text-xs text-amber-300">{actionDisabledReason}</div>
          ) : null}
          {focusedTrack && focusedTrack.identity_id === null ? (
            <div className="mt-3 rounded-lg border border-white/10 bg-slate-900/40 p-3">
              <div className="text-xs font-medium text-slate-200">Manual link</div>
              {identityOptions.length > 0 ? (
                <>
                  <select
                    className="mt-2 w-full rounded-md border border-white/15 bg-slate-950/70 px-2 py-1.5 text-xs text-slate-100"
                    value={linkTargetId}
                    onChange={(event) => setLinkTargetId(event.target.value)}
                  >
                    {identityOptions.map((identity) => (
                      <option key={identity.id} value={identity.id}>
                        {identity.label} (ID {identity.id})
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="mt-2 rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accentSoft disabled:cursor-not-allowed disabled:opacity-45"
                    disabled={!linkTargetId}
                    onClick={runManualLink}
                  >
                    Link Track To Identity
                  </button>
                </>
              ) : (
                <div className="mt-2 text-xs text-slate-400">No identities available to link.</div>
              )}
            </div>
          ) : null}
          {actionState ? <div className="mt-2 text-xs text-accentSoft">{actionState}</div> : null}
        </div>
      </aside>
    </div>
  );
}
