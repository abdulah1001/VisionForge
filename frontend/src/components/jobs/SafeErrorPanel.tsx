import { AlertTriangle } from "lucide-react";
import { ApiError } from "@/api/client";
import { friendlyErrorMessage } from "@/lib/friendlyErrors";

export function SafeErrorPanel({
  error,
  title,
}: {
  error: unknown;
  title?: string;
}) {
  const mapped = friendlyErrorMessage(error);
  const code = error instanceof ApiError ? error.code : mapped.code;
  const message = mapped.message;
  const jobId = error instanceof ApiError ? error.jobId : undefined;
  const heading = title ?? mapped.title;

  return (
    <div
      role="alert"
      className="panel flex gap-3 p-4"
      style={{ borderColor: "var(--danger)" }}
    >
      <AlertTriangle className="shrink-0 text-[var(--danger)]" size={20} aria-hidden />
      <div>
        <strong>{heading}</strong>
        {code && <p className="mono mt-1 text-xs text-[var(--danger)]">{code}</p>}
        <p className="mt-1 text-sm text-[var(--muted)]">{message}</p>
        {jobId && (
          <p className="mono mt-1 text-[10px] text-[var(--dim)]">job {jobId}</p>
        )}
      </div>
    </div>
  );
}
