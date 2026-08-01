import type { JobResult } from "@/api/types";

export function FeatureSummary({ result }: { result: JobResult }) {
  const shape = result.dinov3_feature_shape;
  const identity = result.identity_summary;

  return (
    <section className="panel p-4">
      <h3 className="display text-lg" style={{ color: "var(--feature)" }}>
        DINOv3 features
      </h3>
      <p className="mt-2 text-sm text-[var(--muted)]">
        Local visual embeddings for tracked crops. Numbers are similarity to the reference frame
        when provided—not an identity certainty threshold.
      </p>
      <p className="mono mt-3 text-sm">
        Shape: {shape ? `[${shape.join(", ")}]` : "—"}
      </p>
      {identity && (
        <pre className="mt-3 overflow-x-auto rounded-lg bg-[var(--s2)] p-3 text-[11px] text-[var(--muted)]">
          {JSON.stringify(identity, null, 2)}
        </pre>
      )}
    </section>
  );
}
