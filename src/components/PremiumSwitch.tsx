import { RadioGroup } from "@headlessui/react";
import { motion } from "framer-motion";

export type ViewMode = "live" | "library";

interface PremiumSwitchProps {
  value: ViewMode;
  onChange: (value: ViewMode) => void;
}

const OPTIONS: Array<{ value: ViewMode; label: string }> = [
  { value: "live", label: "Live" },
  { value: "library", label: "Library" }
];

export function PremiumSwitch({ value, onChange }: PremiumSwitchProps): JSX.Element {
  return (
    <RadioGroup value={value} onChange={onChange} aria-label="Choose between live and library views">
      <div className="inline-grid grid-cols-2 gap-1 rounded-xl bg-white/10 p-1 shadow-soft backdrop-blur-md">
        {OPTIONS.map((option) => (
          <RadioGroup.Option key={option.value} value={option.value} className="relative outline-none">
            {({ checked, active }) => (
              <div
                className={`relative rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  checked ? "text-white" : "text-slate-300"
                } ${active ? "ring-2 ring-accent/70 ring-offset-0" : ""}`}
              >
                {checked ? (
                  <motion.span
                    layoutId="premium-switch-thumb"
                    className="absolute inset-0 rounded-lg bg-accent shadow-[0_8px_24px_rgba(124,92,255,0.45)]"
                    transition={{ type: "spring", stiffness: 450, damping: 34, mass: 0.6 }}
                  />
                ) : null}
                <span className="relative z-10">{option.label}</span>
              </div>
            )}
          </RadioGroup.Option>
        ))}
      </div>
    </RadioGroup>
  );
}
