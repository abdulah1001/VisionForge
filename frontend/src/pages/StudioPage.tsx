import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, MousePointer2, Sparkles } from "lucide-react";
import { api, ApiError } from "@/api/client";
import { useCancelJob, useCapabilities, useCreateJob, useJob } from "@/api/hooks";
import { validBox } from "@/api/types";
import { fitProcessingSize } from "@/lib/processingSize";
import { titleCaseLabel } from "@/lib/videoBoxCoords";
import { UploadDropzone } from "@/components/remover/UploadDropzone";
import { VideoOverlay } from "@/components/remover/VideoOverlay";
import { ProcessingScreen } from "@/components/remover/ProcessingScreen";
import { ResultScreen } from "@/components/remover/ResultScreen";
import { RemoverError } from "@/components/remover/RemoverError";
import { FailureScreen } from "@/components/remover/FailureScreen";
import { StudioStepRail } from "@/components/remover/StudioStepRail";
import { pickTracker, useRemoverStore } from "@/store/removerStore";

function formatBytes(n: number): string {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec)) return "—";
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}:${String(r).padStart(2, "0")}` : `${r}s`;
}

export function StudioPage() {
  const phase = useRemoverStore((st) => st.phase);
  const file = useRemoverStore((st) => st.file);
  const videoUrl = useRemoverStore((st) => st.videoUrl);
  const previewId = useRemoverStore((st) => st.previewId);
  const meta = useRemoverStore((st) => st.meta);
  const candidates = useRemoverStore((st) => st.candidates);
  const selectedId = useRemoverStore((st) => st.selectedId);
  const box = useRemoverStore((st) => st.box);
  const selectedLabel = useRemoverStore((st) => st.selectedLabel);
  const selectionMode = useRemoverStore((st) => st.selectionMode);
  const manualDraw = useRemoverStore((st) => st.manualDraw);
  const qualityMode = useRemoverStore((st) => st.qualityMode);
  const anchorTimeSec = useRemoverStore((st) => st.anchorTimeSec);
  const activeJobId = useRemoverStore((st) => st.activeJobId);

  const setPhase = useRemoverStore((st) => st.setPhase);
  const setUpload = useRemoverStore((st) => st.setUpload);
  const setCandidates = useRemoverStore((st) => st.setCandidates);
  const selectObject = useRemoverStore((st) => st.selectObject);
  const setManualBox = useRemoverStore((st) => st.setManualBox);
  const setManualDraw = useRemoverStore((st) => st.setManualDraw);
  const clearSelection = useRemoverStore((st) => st.clearSelection);
  const setQualityMode = useRemoverStore((st) => st.setQualityMode);
  const setAnchorTimeSec = useRemoverStore((st) => st.setAnchorTimeSec);
  const setActiveJob = useRemoverStore((st) => st.setActiveJob);
  const backToDetect = useRemoverStore((st) => st.backToDetect);
  const reset = useRemoverStore((st) => st.reset);

  const caps = useCapabilities();
  const create = useCreateJob();
  const cancel = useCancelJob();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [failedJobId, setFailedJobId] = useState<string | null>(null);
  const [restoredJobId, setRestoredJobId] = useState<string | null>(() =>
    localStorage.getItem("visionforge-active-job"),
  );

  const jobId = activeJobId ?? restoredJobId;
  const pollId =
    phase === "processing" || phase === "result" || Boolean(activeJobId)
      ? jobId
      : null;
  const jobQuery = useJob(pollId ?? "");

  // Restore in-flight / completed job after refresh (only when studio is empty)
  useEffect(() => {
    const failRaw = sessionStorage.getItem("visionforge-last-failure");
    if (failRaw && !file) {
      try {
        const parsed = JSON.parse(failRaw) as { code?: string; message?: string; jobId?: string };
        setError(
          new ApiError(parsed.code ?? "PIPELINE_FAILED", parsed.message ?? "Removal stopped.", 0, parsed.jobId),
        );
        setFailedJobId(parsed.jobId ?? null);
        setPhase("failed");
        return;
      } catch {
        sessionStorage.removeItem("visionforge-last-failure");
      }
    }
    const id = localStorage.getItem("visionforge-active-job");
    if (!id) return;
    const protocol = localStorage.getItem("visionforge-remover-protocol");
    if (protocol !== "full-v2") {
      localStorage.removeItem("visionforge-active-job");
      return;
    }
    if (phase === "processing" || phase === "result" || phase === "failed") return;
    if (file) return;
    setRestoredJobId(id);
    setActiveJob(id);
    setPhase("processing");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Advance to result (or fail) from poll — never depend on whole store object
  useEffect(() => {
    const job = jobQuery.data;
    if (!job || !jobId) return;
    if (
      job.status === "succeeded" ||
      job.status === "partial" ||
      job.status === "review_required"
    ) {
      sessionStorage.removeItem("visionforge-last-failure");
      if (phase !== "result") setPhase("result");
      if (!activeJobId) setActiveJob(job.job_id);
      return;
    }
    if (job.status === "failed" || job.status === "cancelled") {
      const err = new ApiError(
        job.error_code ?? job.status.toUpperCase(),
        job.error_message ?? `Job ${job.status}`,
        0,
        job.job_id,
      );
      setError(err);
      setFailedJobId(job.job_id);
      sessionStorage.setItem(
        "visionforge-last-failure",
        JSON.stringify({
          code: err.code,
          message: err.message,
          jobId: job.job_id,
        }),
      );
      localStorage.removeItem("visionforge-active-job");
      setActiveJob(null);
      setRestoredJobId(null);
      if (phase !== "failed") setPhase("failed");
    }
  }, [
    jobQuery.data,
    jobId,
    phase,
    activeJobId,
    setPhase,
    setActiveJob,
  ]);

  const onFile = useCallback(
    async (next: File) => {
      setError(null);
      setUploading(true);
      setUploadProgress(0);
      try {
        const probe = await api.probeMediaWithProgress(next, setUploadProgress);
        const blobUrl = URL.createObjectURL(next);
        setUpload(
          next,
          blobUrl,
          {
            filename: probe.filename || next.name,
            width: probe.width,
            height: probe.height,
            durationSec: probe.duration_sec,
            fps: probe.fps,
            sizeBytes: next.size,
            estimatedFrames: probe.estimated_frames,
          },
          probe.preview_id,
        );
        setRestoredJobId(null);
      } catch (e) {
        setError(e);
        reset();
      } finally {
        setUploading(false);
        setUploadProgress(null);
      }
    },
    [setUpload, reset],
  );

  const analyze = async () => {
    if (!previewId || !videoRef.current) return;
    setError(null);
    setAnalyzing(true);
    try {
      const video = videoRef.current;
      video.pause();
      const timeSec = Number.isFinite(video.currentTime) ? video.currentTime : 0;
      setAnchorTimeSec(timeSec);
      const res = await api.candidates(previewId, { timeSec });
      const mapped = (res.candidates || []).map((c) => ({
        candidate_id: c.candidate_id,
        box_xyxy: c.box_xyxy as [number, number, number, number],
        score: c.score,
        label: c.label,
      }));
      setCandidates(mapped);
      if (!mapped.length) {
        setError(
          new ApiError(
            "NO_OBJECTS",
            "No objects found on this frame. Try another moment or select manually.",
            0,
          ),
        );
      }
    } catch (e) {
      setError(e);
    } finally {
      setAnalyzing(false);
    }
  };

  const maxSide = qualityMode === "high" ? 960 : 720;
  const proc = useMemo(
    () =>
      meta && meta.width > 0
        ? fitProcessingSize(meta.width, meta.height, maxSide)
        : { width: 0, height: 0 },
    [meta, maxSide],
  );

  const estimatedSeconds = useMemo(() => {
    if (!meta?.durationSec) return null;
    return Math.min(60, meta.durationSec);
  }, [meta]);

  const canRemove =
    Boolean(file) &&
    box != null &&
    meta != null &&
    validBox(box, meta.width, meta.height) &&
    !create.isPending;

  const removeObject = (opts?: { optimized?: boolean }) => {
    if (!file || !box || !meta || (!canRemove && !opts?.optimized)) return;
    setError(null);
    setFailedJobId(null);
    const tracker = pickTracker(caps.data);
    const label = selectedLabel?.trim() || "object";
    const optimized = Boolean(opts?.optimized);
    const side = optimized ? 640 : maxSide;
    const size = fitProcessingSize(meta.width, meta.height, side);
    // Optimized retry: still full duration, but smaller GPU footprint
    const spec = {
      operation: "remove_object" as const,
      tracker,
      box,
      labels: [label],
      selected_label: label,
      anchor_time_sec: anchorTimeSec,
      quality_mode: (optimized ? "standard" : qualityMode) as "standard" | "high",
      mask_confirmed: true as const,
      processing_width: size.width,
      processing_height: size.height,
      max_frames: null,
      analysis_mode: "full" as const,
      selection_mode: selectionMode,
    };
    create.mutate(
      { file, spec },
      {
        onSuccess: (j) => {
          localStorage.setItem("visionforge-active-job", j.job_id);
          localStorage.setItem("visionforge-remover-protocol", "full-v2");
          setActiveJob(j.job_id);
          setPhase("processing");
          setRestoredJobId(j.job_id);
        },
        onError: (e) => {
          setError(e);
          setPhase("failed");
        },
      },
    );
  };

  const onCancel = () => {
    if (!jobId) return;
    cancel.mutate(jobId, {
      onSuccess: () => {
        localStorage.removeItem("visionforge-active-job");
        setActiveJob(null);
        setRestoredJobId(null);
        setError(
          new ApiError("CANCELLED", "Removal was cancelled.", 0, jobId),
        );
        setFailedJobId(jobId);
        setPhase("failed");
      },
      onError: (e) => setError(e),
    });
  };

  const stepIndex =
    phase === "upload"
      ? 0
      : !candidates.length && !box
        ? 1
        : !box
          ? 2
          : 3;

  // Prefer result whenever job finished — even if phase lag
  const showResult =
    Boolean(jobId) &&
    (phase === "result" ||
      ["succeeded", "partial", "review_required"].includes(jobQuery.data?.status ?? ""));

  const showProcessing =
    Boolean(jobId) &&
    phase !== "failed" &&
    !showResult &&
    (phase === "processing" ||
      ["queued", "running", "cancelling"].includes(jobQuery.data?.status ?? ""));

  if (phase === "failed" || (error && !showProcessing && !showResult && failedJobId)) {
    return (
      <>
        <h1 className="sr-only">Removal failed</h1>
        <FailureScreen
          error={error ?? new ApiError("PIPELINE_FAILED", "Removal stopped.", 0)}
          retrying={create.isPending}
          onRetryOptimized={() => {
            if (!file || !box) {
              setPhase("upload");
              setError(null);
              setFailedJobId(null);
              reset();
              return;
            }
            removeObject({ optimized: true });
          }}
          onStartNew={() => {
            localStorage.removeItem("visionforge-active-job");
            sessionStorage.removeItem("visionforge-last-failure");
            setFailedJobId(null);
            setError(null);
            reset();
          }}
        />
      </>
    );
  }

  if (showProcessing && jobId) {
    return (
      <>
        <h1 className="sr-only">Removing object from video</h1>
        {(error || cancel.error) && <RemoverError error={error || cancel.error} />}
        <ProcessingScreen
          stage={jobQuery.data?.stage ?? null}
          percent={jobQuery.data?.overall_percent ?? null}
          onCancel={onCancel}
          cancelling={cancel.isPending || jobQuery.data?.status === "cancelling"}
        />
      </>
    );
  }

  if (showResult && jobId) {
    return (
      <>
        <h1 className="sr-only">Cleaned video result</h1>
        <ResultScreen
          jobId={jobId}
          onRemoveAnother={() => {
            localStorage.removeItem("visionforge-active-job");
            backToDetect();
            setRestoredJobId(null);
          }}
          onStartNew={() => {
            localStorage.removeItem("visionforge-active-job");
            reset();
            setRestoredJobId(null);
          }}
        />
      </>
    );
  }

  return (
    <div className="mx-auto max-w-[1400px] overflow-x-hidden">
      <h1 className="sr-only">Video Object Remover Studio</h1>
      {(error || create.error) && (
        <RemoverError
          error={error || create.error}
          onDismiss={() => setError(null)}
        />
      )}

      <div className="border-b border-[var(--b1)] bg-[color-mix(in_srgb,var(--s1)_55%,transparent)] px-4 py-5 sm:px-6">
        <p className="eyebrow">Object remover</p>
        <h2 className="display mt-1 text-2xl tracking-[-0.04em] sm:text-3xl">
          Remove an object from video
        </h2>
        <p className="mt-2 max-w-xl text-sm text-[var(--dim)]">
          Local GPU pipeline — select clearly, then remove. Original file stays untouched.
        </p>
        <StudioStepRail stepIndex={stepIndex} />
      </div>

      {phase === "upload" || !file || !videoUrl || !meta ? (
        <div className="px-4 py-10 sm:px-6">
          <UploadDropzone
            onFile={(f) => void onFile(f)}
            uploading={uploading}
            progress={uploadProgress}
            successMeta={null}
            onReplace={() => reset()}
          />
        </div>
      ) : (
        <div className="grid gap-4 p-4 lg:grid-cols-[1fr_320px] lg:p-6">
          <section className="min-w-0">
            <div className="studio-stage">
              <video
                ref={videoRef}
                className="mx-auto max-h-[min(70vh,720px)] w-full"
                src={videoUrl}
                controls
                playsInline
                preload="metadata"
                onLoadedMetadata={(e) => {
                  const v = e.currentTarget;
                  if (Number.isFinite(v.duration) && meta.durationSec == null) {
                    /* keep probe meta */
                  }
                }}
              />
              <VideoOverlay
                naturalWidth={meta.width}
                naturalHeight={meta.height}
                candidates={candidates}
                selectedId={selectedId}
                selectedBox={box}
                manualDraw={manualDraw}
                onSelect={(id, b, label) => selectObject(id, b, label)}
                onManualBox={(b) => setManualBox(b)}
              />
            </div>
            <p className="mt-2 text-xs text-[var(--dim)]">
              {meta.filename} · {meta.width}×{meta.height} · {formatDuration(meta.durationSec)} ·{" "}
              {formatBytes(meta.sizeBytes)}
            </p>
          </section>

          <aside className="flex flex-col gap-4">
            <div className="panel p-4">
              <p className="eyebrow">Detect</p>
              <p className="mt-2 text-xs text-[var(--muted)]">
                Pause on the frame where the object is clearest, then analyze.
              </p>
              <button
                type="button"
                className="btn btn-primary mt-3 min-h-11 w-full"
                disabled={analyzing}
                onClick={() => void analyze()}
              >
                <Sparkles size={16} />
                {analyzing ? "Analyzing…" : "Analyze Objects"}
              </button>
              <button
                type="button"
                className="btn mt-2 min-h-11 w-full"
                onClick={() => setManualDraw(!manualDraw)}
              >
                <MousePointer2 size={16} />
                {manualDraw ? "Drawing… click video" : "Select manually"}
              </button>
              {candidates.length > 0 && (
                <ul className="mt-3 max-h-40 space-y-1 overflow-y-auto text-sm">
                  {candidates.map((c) => (
                    <li key={c.candidate_id}>
                      <button
                        type="button"
                        className={`flex w-full items-center justify-between rounded-lg border px-2 py-2 text-left ${
                          selectedId === c.candidate_id
                            ? "border-[var(--tracking)] bg-[color-mix(in_srgb,var(--tracking)_10%,transparent)]"
                            : "border-[var(--b1)]"
                        }`}
                        onClick={() =>
                          selectObject(c.candidate_id, c.box_xyxy, titleCaseLabel(c.label))
                        }
                      >
                        <span>{titleCaseLabel(c.label)}</span>
                        {selectedId === c.candidate_id && <Check size={14} />}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              {box && (
                <div className="mt-3 rounded-lg border border-[var(--success)] px-3 py-2 text-xs text-[var(--success)]">
                  Selected for removal: {selectedLabel || "region"}
                  <button
                    type="button"
                    className="ml-2 underline"
                    onClick={() => clearSelection()}
                  >
                    Change
                  </button>
                </div>
              )}
            </div>

            <div className="panel p-4">
              <p className="eyebrow">Remove</p>
              <p className="mt-2 text-xs text-[var(--muted)]">
                Processes the{" "}
                <strong className="text-[var(--text)]">
                  full video{estimatedSeconds != null ? ` (~${estimatedSeconds.toFixed(0)}s)` : ""}
                </strong>
                , not a 1-second sample. Output ~{proc.width}×{proc.height}.
                Longer / higher quality takes more GPU time.
              </p>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  className={`btn flex-1 ${qualityMode === "standard" ? "btn-primary" : ""}`}
                  onClick={() => setQualityMode("standard")}
                >
                  Standard
                  <span className="block text-[10px] font-normal opacity-80">~720p · safer GPU</span>
                </button>
                <button
                  type="button"
                  className={`btn flex-1 ${qualityMode === "high" ? "btn-primary" : ""}`}
                  onClick={() => setQualityMode("high")}
                >
                  High Quality
                  <span className="block text-[10px] font-normal opacity-80">~960p · needs VRAM</span>
                </button>
              </div>
              <button
                type="button"
                data-testid="remove-object-btn"
                className="btn btn-primary mt-3 min-h-11 w-full"
                disabled={!canRemove}
                onClick={() => removeObject()}
              >
                Remove Object
              </button>
              <button
                type="button"
                className="btn mt-2 min-h-11 w-full"
                onClick={() => {
                  reset();
                  setError(null);
                }}
              >
                Replace video
              </button>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
