import type { JobResult } from "@/api/types";

const DISCLAIMER =
  "These scores express similarity between the visual result and the labels supplied for this job. They are not guaranteed classifications.";

export function SimilarityRanking({
  mobileclip2,
}: {
  mobileclip2?: JobResult["mobileclip2"];
}) {
  const scores = mobileclip2?.mean_scores ?? {};
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => v), 1e-6);

  return (
    <section className="panel p-4">
      <h3 className="display text-lg" style={{ color: "var(--similarity)" }}>
        MobileCLIP2 similarity
      </h3>
      <p className="mt-2 text-sm text-[var(--similarity)]">{DISCLAIMER}</p>
      {mobileclip2?.note && (
        <p className="mt-1 text-xs text-[var(--dim)]">{mobileclip2.note}</p>
      )}
      <dl className="mt-3 grid gap-2 text-xs text-[var(--muted)] sm:grid-cols-3">
        <div>
          <dt>Image features</dt>
          <dd className="mono">
            {mobileclip2?.image_feature_shape
              ? `[${mobileclip2.image_feature_shape.join(", ")}]`
              : "—"}
          </dd>
        </div>
        <div>
          <dt>Text features</dt>
          <dd className="mono">
            {mobileclip2?.text_feature_shape
              ? `[${mobileclip2.text_feature_shape.join(", ")}]`
              : "—"}
          </dd>
        </div>
        <div>
          <dt>Highest ranked</dt>
          <dd>{mobileclip2?.highest_scoring_aggregate_label ?? "—"}</dd>
        </div>
      </dl>
      <ul className="mt-4 space-y-2">
        {entries.map(([label, score]) => (
          <li key={label}>
            <div className="mb-1 flex justify-between text-sm">
              <span>{label}</span>
              <span className="mono text-[var(--dim)]">{score.toFixed(4)}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded bg-[var(--s3)]">
              <div
                className="h-full rounded"
                style={{
                  width: `${(score / max) * 100}%`,
                  background: "var(--similarity)",
                }}
              />
            </div>
          </li>
        ))}
        {!entries.length && (
          <li className="text-sm text-[var(--muted)]">No similarity scores in result.</li>
        )}
      </ul>
    </section>
  );
}
