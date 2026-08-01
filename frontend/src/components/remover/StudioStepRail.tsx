import { Check } from "lucide-react";

const DEFAULT_STEPS = [
  "Upload Video",
  "Analyze Objects",
  "Select Object",
  "Remove Object",
] as const;

export function StudioStepRail({
  stepIndex,
  steps = DEFAULT_STEPS,
}: {
  stepIndex: number;
  steps?: readonly string[];
}) {
  return (
    <ol className="step-rail mt-5" aria-label="Steps">
      {steps.map((label, i) => {
        const done = i < stepIndex;
        const active = i === stepIndex;
        return (
          <li
            key={label}
            className={`step-rail__item ${active ? "is-active" : ""} ${done ? "is-done" : ""}`}
            aria-current={active ? "step" : undefined}
          >
            <span className="step-rail__num" aria-hidden>
              {done ? <Check size={12} strokeWidth={3} /> : i + 1}
            </span>
            {label}
          </li>
        );
      })}
    </ol>
  );
}
