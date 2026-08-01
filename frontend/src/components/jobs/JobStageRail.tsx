import { STAGES } from "@/api/types";

const STAGE_COLOR: Record<string, string> = {
  validating: "var(--muted)",
  preparing_input: "var(--muted)",
  tracking: "var(--tracking)",
  creating_artifacts: "var(--tracking)",
  dinov3: "var(--feature)",
  mobileclip2: "var(--similarity)",
  finalizing: "var(--success)",
  completed: "var(--success)",
};

export function JobStageRail({
  stage,
  status,
}: {
  stage: string | null;
  status: string;
}) {
  const current = stage ?? (status === "succeeded" ? "completed" : null);
  const idx = current ? STAGES.indexOf(current as (typeof STAGES)[number]) : -1;

  return (
    <ol className="flex flex-wrap gap-2" aria-label="Pipeline stages">
      {STAGES.map((s, i) => {
        const done = idx > i || status === "succeeded";
        const active = idx === i && status !== "succeeded";
        return (
          <li
            key={s}
            className="rounded-md border px-2 py-1 text-[11px]"
            style={{
              borderColor: active || done ? STAGE_COLOR[s] ?? "var(--b2)" : "var(--b1)",
              color: active || done ? STAGE_COLOR[s] : "var(--dim)",
              background: active
                ? `color-mix(in srgb, ${STAGE_COLOR[s]} 12%, transparent)`
                : "transparent",
              opacity: done || active ? 1 : 0.55,
            }}
            aria-current={active ? "step" : undefined}
          >
            {s}
          </li>
        );
      })}
    </ol>
  );
}
