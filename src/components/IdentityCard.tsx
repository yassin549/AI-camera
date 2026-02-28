import { motion } from "framer-motion";
import type { IdentitySummary } from "../api/client";
import { AuthenticatedImage } from "./AuthenticatedImage";

interface IdentityCardProps {
  identity: IdentitySummary;
  layoutId: string;
  onOpen: (identity: IdentitySummary, layoutId: string) => void;
}

export function IdentityCard({ identity, layoutId, onOpen }: IdentityCardProps): JSX.Element {
  const thumb = identity.face_samples[0] ?? identity.body_samples[0] ?? "";
  const frequency = identity.stats?.frequency ?? identity.stats?.sightings ?? 0;

  return (
    <motion.button
      layoutId={layoutId}
      type="button"
      className="group glass-panel relative h-[11.5rem] w-[11.5rem] overflow-hidden rounded-2xl border border-white/20 bg-slate-950/50 text-left transition-[border-color,box-shadow] duration-150 hover:border-accentSoft hover:shadow-soft focus-visible:border-accentSoft"
      onClick={() => onOpen(identity, layoutId)}
      aria-label={`Open identity ${identity.id}`}
      data-testid={`identity-card-${identity.id}`}
    >
      {thumb ? (
        <AuthenticatedImage
          src={thumb}
          alt={`Identity ${identity.id}`}
          loading="lazy"
          className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.04]"
        />
      ) : (
        <div className="grid h-full w-full place-items-center bg-slate-900/80 text-xs uppercase tracking-[0.2em] text-slate-400">
          No sample
        </div>
      )}

      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/95 via-slate-950/70 to-transparent p-2.5">
        <div className="truncate text-xs font-semibold text-slate-100">ID {identity.id}</div>
        <div className="truncate text-[11px] text-slate-300">Freq: {frequency}</div>
      </div>
    </motion.button>
  );
}
