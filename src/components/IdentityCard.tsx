import { motion } from "framer-motion";
import type { IdentitySummary } from "../api/client";

interface IdentityCardProps {
  identity: IdentitySummary;
  layoutId: string;
  onOpen: (identity: IdentitySummary, layoutId: string) => void;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function IdentityCard({ identity, layoutId, onOpen }: IdentityCardProps): JSX.Element {
  const thumb = identity.face_samples[0] ?? identity.body_samples[0] ?? "";
  const frequency = identity.stats?.frequency ?? identity.stats?.sightings ?? 0;

  return (
    <motion.button
      layoutId={layoutId}
      type="button"
      className="group glass-panel flex w-full flex-col overflow-hidden rounded-2xl text-left"
      onClick={() => onOpen(identity, layoutId)}
      aria-label={`Open identity ${identity.id}`}
      data-testid={`identity-card-${identity.id}`}
      whileHover={{ y: -2 }}
      transition={{ duration: 0.18 }}
    >
      <div className="relative aspect-video bg-slate-900/80">
        {thumb ? (
          <img
            src={thumb}
            alt={`Identity ${identity.id}`}
            loading="lazy"
            className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.01]"
          />
        ) : (
          <div className="grid h-full place-items-center text-xs uppercase tracking-[0.2em] text-slate-400">
            No sample
          </div>
        )}
        <span className="absolute right-3 top-3 rounded-full bg-slate-950/70 px-3 py-1 text-xs text-slate-200">
          ID {identity.id}
        </span>
      </div>
      <div className="space-y-2 p-4">
        <div className="text-sm font-semibold text-slate-100">Identity {identity.id}</div>
        <div className="text-xs text-slate-300">First seen: {formatDate(identity.first_seen)}</div>
        <div className="text-xs text-slate-300">Last seen: {formatDate(identity.last_seen)}</div>
        <div className="text-xs text-accentSoft">Frequency: {frequency}</div>
      </div>
    </motion.button>
  );
}
