import { Link } from "react-router-dom";
import { VisionForgeMark } from "@/assets/logo";

export function VisionForgeLogo({ compact = false }: { compact?: boolean }) {
  return (
    <Link to="/" className="group flex items-center gap-2.5" aria-label="VisionForge home">
      <VisionForgeMark className="h-8 w-8 text-[var(--text)] transition group-hover:text-[var(--tracking)]" />
      {!compact && (
        <span className="display text-lg font-semibold tracking-[-0.04em]">
          Vision<span className="text-[var(--tracking)]">Forge</span>
        </span>
      )}
    </Link>
  );
}
