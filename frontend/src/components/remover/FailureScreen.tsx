import { AlertTriangle, RotateCcw, Shrink } from "lucide-react";
import { friendlyErrorMessage } from "@/lib/friendlyErrors";

export function FailureScreen({
  error,
  onRetryOptimized,
  onStartNew,
  retrying,
}: {
  error: unknown;
  onRetryOptimized: () => void;
  onStartNew: () => void;
  retrying?: boolean;
}) {
  const { title, message, code } = friendlyErrorMessage(error);
  const raw =
    error && typeof error === "object" && "message" in error
      ? String((error as { message?: unknown }).message ?? "")
      : "";
  const oom =
    code === "GPU_OOM" ||
    /gpu|memory|oom|out of memory/i.test(message) ||
    /out of memory|outofmemory|cuda/i.test(raw);

  return (
    <div className="mx-auto flex min-h-[calc(100vh-68px)] max-w-lg flex-col justify-center px-4 py-12 sm:px-6">
      <div className="panel p-7" style={{ borderColor: "var(--danger)" }}>
        <div className="flex gap-3">
          <AlertTriangle className="shrink-0 text-[var(--danger)]" size={22} aria-hidden />
          <div>
            <p className="eyebrow" style={{ color: "var(--danger)" }}>
              Removal stopped
            </p>
            <h1 className="display mt-2 text-3xl tracking-[-0.04em]">{title}</h1>
            <p className="mt-2 text-sm text-[var(--muted)]">{message}</p>
            {code && <p className="mono mt-2 text-[10px] text-[var(--dim)]">{code}</p>}
            {oom && (
              <p className="mt-3 text-sm text-[var(--warning)]">
                Your GPU ran out of memory on the full clip. Retry Optimized (~640p)
                uses less VRAM — original video is unchanged.
              </p>
            )}
          </div>
        </div>
        <div className="mt-6 flex flex-wrap gap-3">
          {oom && (
            <button
              type="button"
              className="btn btn-primary min-h-11"
              disabled={retrying}
              onClick={onRetryOptimized}
            >
              <Shrink size={16} />
              {retrying ? "Starting…" : "Retry Optimized"}
            </button>
          )}
          <button type="button" className="btn min-h-11" onClick={onStartNew}>
            <RotateCcw size={16} />
            Start New Video
          </button>
        </div>
      </div>
    </div>
  );
}
