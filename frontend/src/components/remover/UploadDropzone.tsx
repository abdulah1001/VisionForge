import { useCallback, useRef, useState } from "react";
import { CheckCircle2, Film, Upload } from "lucide-react";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
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

export interface UploadMeta {
  filename: string;
  width: number;
  height: number;
  durationSec: number | null;
  sizeBytes: number;
}

export function UploadDropzone({
  onFile,
  progress,
  uploading,
  successMeta,
  onReplace,
}: {
  onFile: (file: File) => void;
  progress: number | null;
  uploading: boolean;
  successMeta: UploadMeta | null;
  onReplace: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const accept = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!/\.(mp4|mov|webm|avi|mkv)$/i.test(file.name) && !file.type.startsWith("video/")) {
        return;
      }
      onFile(file);
    },
    [onFile],
  );

  if (successMeta && !uploading) {
    return (
      <div className="upload-success mx-auto max-w-xl rounded-[var(--radius)] border border-[var(--success)] bg-[color-mix(in_srgb,var(--success)_8%,var(--s1))] p-7 shadow-[var(--shadow-soft)]">
        <div className="flex items-start gap-3">
          <CheckCircle2 className="mt-0.5 shrink-0 text-[var(--success)]" size={22} />
          <div className="min-w-0 flex-1">
            <p className="font-semibold text-[var(--text)]">Video ready</p>
            <p className="mt-1 truncate text-sm text-[var(--muted)]">{successMeta.filename}</p>
            <dl className="mt-4 grid grid-cols-3 gap-3 text-xs text-[var(--dim)]">
              <div>
                <dt>Duration</dt>
                <dd className="mono mt-1 text-[var(--muted)]">
                  {formatDuration(successMeta.durationSec)}
                </dd>
              </div>
              <div>
                <dt>Resolution</dt>
                <dd className="mono mt-1 text-[var(--muted)]">
                  {successMeta.width}×{successMeta.height}
                </dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd className="mono mt-1 text-[var(--muted)]">
                  {formatBytes(successMeta.sizeBytes)}
                </dd>
              </div>
            </dl>
            <button type="button" className="btn mt-5 min-h-11" onClick={onReplace}>
              Replace video
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`upload-dropzone mx-auto flex max-w-xl flex-col items-center justify-center rounded-[var(--radius)] px-6 py-16 text-center shadow-[var(--shadow-soft)] ${
        dragOver ? "is-drag" : ""
      } ${uploading ? "is-uploading" : ""}`}
      onDragEnter={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        accept(e.dataTransfer.files?.[0]);
      }}
    >
      <div className="mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-[color-mix(in_srgb,var(--tracking)_12%,var(--s3))] text-[var(--tracking)]">
        <Film size={26} aria-hidden />
      </div>
      <p className="display text-2xl">Upload a video</p>
      <p className="mt-2 max-w-sm text-sm text-[var(--dim)]">
        Drag and drop, or browse. MP4, MOV, WebM, or AVI — up to ~60 seconds recommended.
      </p>
      <button
        type="button"
        className="btn btn-primary mt-6 min-h-11"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
      >
        <Upload size={16} />
        {uploading ? "Uploading…" : "Browse files"}
      </button>
      <input
        ref={inputRef}
        hidden
        type="file"
        accept="video/*,.mp4,.mov,.webm,.avi,.mkv"
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {uploading && (
        <div
          className="mt-7 w-full max-w-xs"
          role="progressbar"
          aria-valuenow={progress ?? 0}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${progress ?? 8}%` }}
            />
          </div>
          <p className="mono mt-2 text-xs text-[var(--dim)]">
            {progress != null ? `${progress}%` : "Uploading…"}
          </p>
        </div>
      )}
    </div>
  );
}
