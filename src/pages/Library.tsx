import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getIdentities, type IdentitySummary } from "../api/client";
import { IdentityCard } from "../components/IdentityCard";

interface LibraryProps {
  initialIdentities?: IdentitySummary[];
}

type SortMode = "last_seen" | "frequency";

export function Library({ initialIdentities }: LibraryProps): JSX.Element {
  const navigate = useNavigate();
  const [identities, setIdentities] = useState<IdentitySummary[]>(initialIdentities ?? []);
  const [loading, setLoading] = useState<boolean>(!initialIdentities);
  const [error, setError] = useState<string>("");
  const [search, setSearch] = useState<string>("");
  const [sortBy, setSortBy] = useState<SortMode>("last_seen");

  useEffect(() => {
    if (initialIdentities) {
      return;
    }
    let cancelled = false;
    const load = async (silent = false) => {
      if (!silent) {
        setLoading(true);
      }
      try {
        const payload = await getIdentities();
        if (!cancelled) {
          setIdentities(payload);
          setError("");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load identities");
        }
      } finally {
        if (!cancelled && !silent) {
          setLoading(false);
        }
      }
    };

    load(false);
    const interval = window.setInterval(() => {
      load(true);
    }, 4000);
    const handleWake = () => {
      if (document.visibilityState === "visible") {
        load(true);
      }
    };
    window.addEventListener("focus", handleWake);
    document.addEventListener("visibilitychange", handleWake);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
      window.removeEventListener("focus", handleWake);
      document.removeEventListener("visibilitychange", handleWake);
    };
  }, [initialIdentities]);

  const filtered = useMemo(() => {
    const normalized = search.trim().toLowerCase();
    const base = identities.filter((identity) => {
      if (!normalized) {
        return true;
      }
      return (
        identity.id.toLowerCase().includes(normalized) ||
        identity.first_seen.toLowerCase().includes(normalized) ||
        identity.last_seen.toLowerCase().includes(normalized)
      );
    });

    return [...base].sort((a, b) => {
      if (sortBy === "frequency") {
        const fa = a.stats?.frequency ?? a.stats?.sightings ?? 0;
        const fb = b.stats?.frequency ?? b.stats?.sightings ?? 0;
        return fb - fa;
      }
      return new Date(b.last_seen).getTime() - new Date(a.last_seen).getTime();
    });
  }, [identities, search, sortBy]);

  const handleOpenIdentity = (identity: IdentitySummary, layoutId: string) => {
    navigate(`/library/${identity.id}`, {
      state: {
        identity,
        layoutId
      }
    });
  };

  return (
    <section className="h-full overflow-auto px-4 pb-4 pt-2">
      <div className="glass-panel rounded-2xl p-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-lg font-semibold text-slate-100">Identities Library</h1>
            <p className="text-sm text-slate-400">Search, sort, and inspect recognized identities.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search identity..."
              aria-label="Search identities"
              className="rounded-lg border border-white/15 bg-slate-950/40 px-3 py-2 text-sm text-slate-100 outline-none ring-accent focus:ring-2"
            />
            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value as SortMode)}
              aria-label="Sort identities"
              className="rounded-lg border border-white/15 bg-slate-950/40 px-3 py-2 text-sm text-slate-100 outline-none ring-accent focus:ring-2"
            >
              <option value="last_seen">Last seen</option>
              <option value="frequency">Frequency</option>
            </select>
          </div>
        </div>

        {loading ? <div className="mt-6 text-sm text-slate-400">Loading identities...</div> : null}
        {error ? <div className="mt-6 text-sm text-rose-300">{error}</div> : null}

        <motion.div
          layout
          className="mt-6 grid justify-center gap-4"
          style={{ gridTemplateColumns: "repeat(auto-fit, minmax(11.5rem, 11.5rem))" }}
          initial={false}
        >
          {filtered.map((identity) => {
            const layoutId = `identity-card-${identity.id}`;
            return (
              <IdentityCard
                key={identity.id}
                identity={identity}
                layoutId={layoutId}
                onOpen={handleOpenIdentity}
              />
            );
          })}
        </motion.div>
      </div>
    </section>
  );
}
