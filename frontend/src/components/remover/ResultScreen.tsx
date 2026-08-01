import { useEffect, useRef, useState } from "react";
import { Download, RotateCcw, Sparkles, AlertTriangle } from "lucide-react";
import { api } from "@/api/client";

const SHORT_SAMPLE_SEC = 2.5;

export function ResultScreen({
  jobId,
  onRemoveAnother,
  onStartNew,
}: {
  jobId: string;
  onRemoveAnother: () => void;
  onStartNew: () => void;
}) {
  const src = `${api.cleanedUrl(jobId)}?t=${encodeURIComponent(jobId)}`;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [durationSec, setDurationSec] = useState<number | null>(null);

  useEffect(() => {
    setLoadError(null);
    setReady(false);
    setDurationSec(null);
  }, [src]);

  const shortSample =
    durationSec != null &&
    Number.isFinite(durationSec) &&
    durationSec > 0 &&
    durationSec < SHORT_SAMPLE_SEC;

  return (
    <div className="mx-auto flex min-h-[calc(100vh-68px)] max-w-5xl flex-col px-4 py-10 sm:px-6">
      <p className="eyebrow reveal">Complete</p>
      <h1 className="display reveal reveal-delay-1 mt-2 text-3xl sm:text-5xl">
        Your cleaned video
      </h1>
      <p className="reveal reveal-delay-2 mt-3 max-w-xl text-sm text-[var(--muted)]">
        Preview plays below
        {durationSec != null && Number.isFinite(durationSec)
          ? ` · ${durationSec.toFixed(1)}s`
          : ""}
        . Download anytime — the original file was never modified.
      </p>

      {shortSample && (
        <div
          className="panel mt-5 flex gap-3 p-4"
          style={{ borderColor: "var(--warning)" }}
          role="status"
        >
          <AlertTriangle className="shrink-0 text-[var(--warning)]" size={20} aria-hidden />
          <div className="text-sm">
            <strong>Short sample (~{durationSec?.toFixed(1)}s)</strong>
            <p className="mt-1 text-[var(--muted)]">
              This result is from an older short test run. Click{" "}
              <strong>Start New Video</strong>, upload again, then Remove — full clip
              (up to ~60s) will process now.
            </p>
          </div>
        </div>
      )}

      <div className="result-frame reveal reveal-delay-2 mt-8">
        {!ready && !loadError && (
          <div className="absolute inset-0 z-10 grid place-items-center bg-black/55 text-sm text-[var(--muted)]">
            Loading preview…
          </div>
        )}
        <video
          ref={videoRef}
          key={src}
          className="mx-auto max-h-[min(75vh,820px)] w-full bg-black"
          src={src}
          controls
          playsInline
          autoPlay
          preload="auto"
          onLoadedMetadata={(e) => {
            const d = e.currentTarget.duration;
            if (Number.isFinite(d) && d > 0) setDurationSec(d);
          }}
          onLoadedData={() => setReady(true)}
          onCanPlay={() => setReady(true)}
          onError={() =>
            setLoadError("Preview couldn’t load. Wait a moment or use Download.")
          }
        />
      </div>

      {loadError && (
        <p className="mt-3 text-sm text-[var(--warning)]" role="alert">
          {loadError}
        </p>
      )}

      <div className="mt-7 flex flex-wrap gap-3">
        <button type="button" className="btn btn-primary min-h-11" onClick={onStartNew}>
          <RotateCcw size={16} />
          Start New Video
        </button>
        <button type="button" className="btn min-h-11" onClick={onRemoveAnother}>
          <Sparkles size={16} />
          Remove Another Object
        </button>
        <a className="btn min-h-11" href={src} download={`cleaned-${jobId}.mp4`}>
          <Download size={16} />
          Download Video
        </a>
      </div>
    </div>
  );
}
