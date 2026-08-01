import { friendlyProcessingStage } from "@/lib/videoBoxCoords";

const STAGES = [
  { key: "preparing", label: "Prepare" },
  { key: "tracking", label: "Track" },
  { key: "inpainting", label: "Rebuild" },
  { key: "encoding", label: "Encode" },
] as const;

function stageIndex(stage: string | null): number {
  const s = (stage || "").toLowerCase();
  if (s.includes("encod") || s.includes("final") || s.includes("complet")) return 3;
  if (s.includes("inpaint") || s.includes("propaint") || s.includes("rebuild")) return 2;
  if (s.includes("track") || s.includes("mask")) return 1;
  if (s.includes("prepar") || s.includes("valid") || s.includes("extract")) return 0;
  return 0;
}

export function ProcessingScreen({
  stage,
  percent,
  onCancel,
  cancelling,
}: {
  stage: string | null;
  percent: number | null;
  onCancel: () => void;
  cancelling?: boolean;
}) {
  const label = friendlyProcessingStage(stage);
  const pct = Math.max(0, Math.min(100, Math.round(percent ?? 0)));
  const active = stageIndex(stage);

  return (
    <div className="processing-screen flex min-h-[calc(100vh-68px)] flex-col items-center justify-center px-6 py-16">
      <div className="pointer-events-none absolute inset-0 grid-bg opacity-25" aria-hidden />
      <div className="processing-orb" aria-hidden />
      <p className="eyebrow">Removing object</p>
      <h1 className="display mt-3 text-center text-3xl sm:text-4xl">{label}</h1>
      <p className="mt-3 max-w-md text-center text-sm text-[var(--dim)]">
        Stay on this page — your video is processed locally on this machine.
      </p>

      <ol className="mt-8 flex flex-wrap justify-center gap-2" aria-label="Pipeline stages">
        {STAGES.map((s, i) => (
          <li
            key={s.key}
            className={`rounded-full border px-3 py-1.5 text-[11px] font-semibold tracking-wide ${
              i === active
                ? "border-[var(--tracking)] text-[var(--tracking)] shadow-[var(--glow)]"
                : i < active
                  ? "border-[var(--success)] text-[var(--success)]"
                  : "border-[var(--b1)] text-[var(--dim)]"
            }`}
          >
            {s.label}
          </li>
        ))}
      </ol>

      <div
        className="mt-10 w-full max-w-md"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Overall progress"
      >
        <div className="flex justify-between text-xs text-[var(--dim)]">
          <span>Progress</span>
          <span className="mono">{pct}%</span>
        </div>
        <div className="progress-track mt-2">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <button
        type="button"
        className="btn mt-10 min-h-11"
        disabled={cancelling}
        onClick={onCancel}
      >
        {cancelling ? "Cancelling…" : "Cancel"}
      </button>
    </div>
  );
}
