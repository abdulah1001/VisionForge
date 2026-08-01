import { AlertTriangle } from "lucide-react";
import { friendlyErrorMessage } from "@/lib/friendlyErrors";

export function RemoverError({
  error,
  onDismiss,
}: {
  error: unknown;
  onDismiss?: () => void;
}) {
  const { title, message, code } = friendlyErrorMessage(error);
  return (
    <div
      role="alert"
      className="panel mx-4 mt-4 flex gap-3 p-4 sm:mx-6"
      style={{ borderColor: "var(--danger)" }}
    >
      <AlertTriangle className="shrink-0 text-[var(--danger)]" size={20} aria-hidden />
      <div className="min-w-0 flex-1">
        <strong>{title}</strong>
        <p className="mt-1 text-sm text-[var(--muted)]">{message}</p>
        {code && <p className="mono mt-1 text-[10px] text-[var(--dim)]">{code}</p>}
      </div>
      {onDismiss && (
        <button type="button" className="btn btn-ghost min-h-11 shrink-0" onClick={onDismiss}>
          Dismiss
        </button>
      )}
    </div>
  );
}
