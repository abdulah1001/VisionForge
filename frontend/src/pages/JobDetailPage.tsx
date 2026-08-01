import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  useArtifacts,
  useCancelJob,
  useJob,
  useResult,
} from "@/api/hooks";
import { api, ApiError } from "@/api/client";
import { isTerminal } from "@/api/types";
import { JobStageRail } from "@/components/jobs/JobStageRail";
import { SafeErrorPanel } from "@/components/jobs/SafeErrorPanel";
import { ResultViewer } from "@/components/jobs/ResultViewer";
import { SimilarityRanking } from "@/components/jobs/SimilarityRanking";
import { FeatureSummary } from "@/components/jobs/FeatureSummary";
import { ArtifactBrowser } from "@/components/jobs/ArtifactBrowser";

function elapsed(start: string | null, end: string | null): string {
  if (!start) return "—";
  const a = Date.parse(start);
  const b = end ? Date.parse(end) : Date.now();
  if (!Number.isFinite(a) || !Number.isFinite(b)) return "—";
  return `${Math.max(0, (b - a) / 1000).toFixed(1)}s`;
}

export function JobDetailPage() {
  const { jobId = "" } = useParams();
  const jobQ = useJob(jobId);
  const job = jobQ.data;
  const showResults = ["succeeded", "review_required", "partial", "failed"].includes(
    job?.status ?? "",
  );
  const resultQ = useResult(jobId, showResults);
  const artsQ = useArtifacts(jobId, showResults || job?.status === "cancelled");
  const cancel = useCancelJob();
  const [confirmCancel, setConfirmCancel] = useState(false);

  useEffect(() => {
    if (job && isTerminal(job.status)) {
      const active = localStorage.getItem("visionforge-active-job");
      if (active === job.job_id) localStorage.removeItem("visionforge-active-job");
    }
  }, [job]);

  const annotated = useMemo(() => {
    const list = artsQ.data ?? [];
    return list.find((a) => a.name === "annotated.mp4" || a.path.endsWith("annotated.mp4"));
  }, [artsQ.data]);

  if (jobQ.isLoading) return <p className="p-8 text-[var(--muted)]">Loading job…</p>;
  if (jobQ.error) {
    return (
      <div className="p-8">
        <SafeErrorPanel
          error={
            jobQ.error instanceof ApiError
              ? jobQ.error
              : new ApiError("JOB_LOAD", String(jobQ.error), 0)
          }
        />
      </div>
    );
  }
  if (!job) return <p className="p-8">Job not found.</p>;
  const result = resultQ.data;

  return (
    <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-6 lg:px-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Job detail</p>
          <h1 className="display mt-1 text-2xl">
            <span className="mono text-base text-[var(--muted)]">{job.job_id}</span>
          </h1>
          <p className="mt-2 text-sm text-[var(--muted)]">
            Tracker <strong>{job.tracker}</strong>
            {job.runtime ? ` · ${job.runtime}` : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link className="btn" to="/jobs">
            All jobs
          </Link>
          {(job.status === "queued" ||
            job.status === "running" ||
            job.status === "cancelling") && (
            <button
              type="button"
              className="btn"
              style={{ borderColor: "var(--danger)", color: "var(--danger)" }}
              disabled={cancel.isPending || job.status === "cancelling"}
              onClick={() => setConfirmCancel(true)}
            >
              {job.status === "cancelling" ? "Cancelling…" : "Cancel"}
            </button>
          )}
          {showResults && (
            <a className="btn btn-primary" href={api.downloadUrl(job.job_id)}>
              Download ZIP
            </a>
          )}
        </div>
      </header>

      <section className="panel p-4">
        <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
          <StatusPill status={job.status} />
          {job.queue_position != null && job.status === "queued" && (
            <span className="text-[var(--warning)]">
              Queue position {job.queue_position} · one GPU job at a time
            </span>
          )}
          {job.overall_percent != null && (
            <span className="mono text-[var(--dim)]">{job.overall_percent}%</span>
          )}
          <span className="mono text-[var(--dim)]">
            {job.frames_completed ?? "—"}/{job.frames_total ?? "—"} ·{" "}
            {elapsed(job.started_at ?? job.created_at, job.finished_at)}
          </span>
        </div>
        <JobStageRail stage={job.stage} status={job.status} />
        <div className="sr-only" aria-live="polite">
          Job {job.status}
          {job.stage ? `, stage ${job.stage}` : ""}
        </div>
      </section>

      {job.status === "failed" && (
        <SafeErrorPanel
          error={
            new ApiError(
              job.error_code ?? "JOB_FAILED",
              job.error_message ?? "Job failed",
              0,
              job.job_id,
            )
          }
        />
      )}

      {job.status === "cancelled" && (
        <div className="panel p-4 text-sm" style={{ borderColor: "var(--warning)" }}>
          <strong style={{ color: "var(--warning)" }}>Cancelled</strong>
          <p className="mt-1 text-[var(--muted)]">
            Partial artifacts may remain for diagnosis. This is not a failure.
          </p>
        </div>
      )}

      {showResults && result && (
        <>
          {(job.status === "review_required" || job.status === "partial") && (
            <div
              className="panel p-4 text-sm"
              style={{ borderColor: "var(--warning)" }}
            >
              <strong style={{ color: "var(--warning)" }}>
                {job.status === "review_required"
                  ? "Review required"
                  : "Partial result"}
              </strong>
              <p className="mt-1 text-[var(--muted)]">
                Tracking produced usable output, but quality checks found invalid masks,
                empty masks, or suspected drift. Inspect the annotated video before trusting
                the sequence.
              </p>
            </div>
          )}

          <section className="panel p-4">
            <h2 className="display text-xl">Annotated video</h2>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Primary product result — selected object marked across processed frames.
            </p>
            {annotated ? (
              <video
                className="mt-4 max-h-[520px] w-full rounded-xl border border-[var(--b1)] bg-black"
                controls
                playsInline
                src={api.artifactUrl(job.job_id, annotated.id)}
              >
                Annotated tracking video
              </video>
            ) : (
              <p className="mt-4 text-sm text-[var(--warning)]">
                Annotated MP4 not listed yet. Check artifacts or re-run if encoding failed.
              </p>
            )}
            {result.annotated_video && (
              <p className="mono mt-2 text-xs text-[var(--dim)]">
                {result.annotated_video.width}×{result.annotated_video.height} ·{" "}
                {result.annotated_video.frames} frames ·{" "}
                {result.annotated_video.duration_sec}s · {result.annotated_video.codec}
                {result.annotated_video.audio
                  ? ` · audio_present=${String(result.annotated_video.audio.audio_present)} · preserved=${String(result.annotated_video.audio.audio_preserved)}`
                  : ""}
                {result.annotated_video.audio_note
                  ? ` · ${result.annotated_video.audio_note}`
                  : ""}
              </p>
            )}
            {annotated && (
              <a
                className="btn mt-3 inline-flex min-h-11"
                href={api.artifactUrl(job.job_id, annotated.id)}
                download="annotated.mp4"
              >
                Download MP4
              </a>
            )}
          </section>

          <section className="panel grid gap-4 p-4 md:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="Valid / invalid masks"
              value={`${result.frames?.valid_masks ?? result.artifact_counts?.valid_masks ?? "—"} / ${result.frames?.invalid_masks ?? result.artifact_counts?.invalid_masks ?? "—"}`}
            />
            <Stat
              label="Empty masks"
              value={String(result.frames?.empty_masks ?? result.artifact_counts?.empty_masks ?? "—")}
            />
            <Stat
              label="Drift warnings"
              value={String(result.quality?.suspected_drift_count ?? "—")}
            />
            <Stat
              label="Fallback"
              value={result.mock_or_fallback_used ? "USED" : "None"}
            />
          </section>

          {result.quality?.frames_requiring_review?.length ? (
            <section className="panel p-4">
              <h3 className="display text-lg">Quality review</h3>
              <p className="mt-1 text-xs text-[var(--muted)]">
                {result.quality.note}
              </p>
              <p className="mono mt-2 text-xs">
                Frames: {result.quality.frames_requiring_review.slice(0, 40).join(", ")}
                {(result.quality.frames_requiring_review.length ?? 0) > 40 ? "…" : ""}
              </p>
              <ManualCorrectionPanel
                jobId={job.job_id}
                tracker={job.tracker}
                reviewFrames={result.quality.frames_requiring_review}
                defaultBox={
                  (result as { bounding_box_original_xyxy?: number[] })
                    .bounding_box_original_xyxy as
                    | [number, number, number, number]
                    | undefined
                }
              />
            </section>
          ) : null}

          <ResultViewer result={result} />
          <FeatureSummary result={result} />
          <SimilarityRanking mobileclip2={result.mobileclip2} />
          <ArtifactBrowser jobId={job.job_id} artifacts={artsQ.data ?? []} />
        </>
      )}

      {confirmCancel && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="cancel-title"
        >
          <div className="panel max-w-md p-5">
            <h2 id="cancel-title" className="display text-xl">
              Cancel this job?
            </h2>
            <p className="mt-2 text-sm text-[var(--muted)]">
              Partial artifacts may remain. Status will be cancelled, not failed.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button type="button" className="btn" onClick={() => setConfirmCancel(false)}>
                Keep running
              </button>
              <button
                type="button"
                className="btn btn-primary"
                disabled={cancel.isPending}
                onClick={() =>
                  cancel.mutate(job.job_id, { onSettled: () => setConfirmCancel(false) })
                }
              >
                Confirm cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const color =
    status === "succeeded"
      ? "var(--success)"
      : status === "failed"
        ? "var(--danger)"
        : status === "review_required" || status === "partial"
          ? "var(--warning)"
          : status === "cancelled" || status === "cancelling"
            ? "var(--dim)"
            : "var(--tracking)";
  return (
    <span className="status-pill" style={{ borderColor: color, color }}>
      {status.replaceAll("_", " ")}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-[var(--dim)]">{label}</p>
      <p className="mono mt-1 text-sm">{value}</p>
    </div>
  );
}

function ManualCorrectionPanel({
  jobId,
  tracker,
  reviewFrames,
  defaultBox,
}: {
  jobId: string;
  tracker: string;
  reviewFrames: number[];
  defaultBox?: [number, number, number, number];
}) {
  const navigate = useNavigate();
  const [frameIndex, setFrameIndex] = useState(reviewFrames[0] ?? 0);
  const [box, setBox] = useState<[number, number, number, number]>(
    defaultBox ?? [40, 40, 200, 200],
  );
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  return (
    <div className="mt-4 space-y-3 border-t border-[var(--b1)] pt-4">
      <p className="text-sm font-medium">Manual correction / reinitialization</p>
      <p className="text-xs text-[var(--muted)]">
        Creates a child revision job from the selected frame. Parent artifacts are preserved.
      </p>
      <label className="block text-xs text-[var(--dim)]">
        Frame index
        <input
          type="number"
          min={0}
          className="mt-1 w-full rounded border border-[var(--b1)] bg-[var(--s2)] px-2 py-2"
          value={frameIndex}
          onChange={(e) => setFrameIndex(Number(e.target.value))}
        />
      </label>
      <div className="grid grid-cols-2 gap-2">
        {(["x1", "y1", "x2", "y2"] as const).map((name, i) => (
          <label key={name} className="mono text-[10px] text-[var(--dim)]">
            {name}
            <input
              type="number"
              className="mt-1 w-full rounded border border-[var(--b1)] bg-[var(--s2)] px-2 py-2 text-sm"
              value={Math.round(box[i]!)}
              onChange={(e) => {
                const next = [...box] as [number, number, number, number];
                next[i] = Number(e.target.value);
                setBox(next);
              }}
            />
          </label>
        ))}
      </div>
      {err && <p className="text-xs text-[var(--danger)]">{err}</p>}
      <button
        type="button"
        className="btn btn-primary min-h-11"
        disabled={busy}
        onClick={() => {
          setBusy(true);
          setErr(null);
          void api
            .submitCorrection(jobId, {
              frame_index: frameIndex,
              box,
              tracker,
              mask_confirmed: true,
              analysis_mode: "full",
            })
            .then((j) => {
              localStorage.setItem("visionforge-active-job", j.job_id);
              void navigate(`/jobs/${j.job_id}`);
            })
            .catch((e) => setErr(String(e)))
            .finally(() => setBusy(false));
        }}
      >
        {busy ? "Submitting…" : "Submit correction revision"}
      </button>
    </div>
  );
}
