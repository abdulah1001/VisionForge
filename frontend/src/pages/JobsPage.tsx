import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useJobs } from "@/api/hooks";
import { api } from "@/api/client";
import type { JobStatus, Tracker } from "@/api/types";

type Filter = "all" | JobStatus | Tracker;

function statusColor(status: string): string {
  if (status === "succeeded") return "var(--success)";
  if (status === "failed") return "var(--danger)";
  if (status === "review_required" || status === "partial") return "var(--warning)";
  if (status === "cancelled" || status === "cancelling") return "var(--dim)";
  return "var(--tracking)";
}

export function JobsPage() {
  const { data: jobs = [], isLoading, error } = useJobs();
  const [filter, setFilter] = useState<Filter>("all");

  const filtered = useMemo(() => {
    return jobs.filter((j) => {
      if (filter === "all") return true;
      if (filter === "edgetam" || filter === "sam31") return j.tracker === filter;
      return j.status === filter;
    });
  }, [jobs, filter]);

  const filters: { id: Filter; label: string }[] = [
    { id: "all", label: "All" },
    { id: "queued", label: "Queued" },
    { id: "running", label: "Running" },
    { id: "succeeded", label: "Succeeded" },
    { id: "review_required", label: "Review" },
    { id: "failed", label: "Failed" },
    { id: "cancelled", label: "Cancelled" },
    { id: "edgetam", label: "EdgeTAM" },
    { id: "sam31", label: "SAM 3.1" },
  ];

  return (
    <div className="mx-auto max-w-[1200px] px-4 py-8 lg:px-8">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">History</p>
          <h1 className="display mt-1 text-4xl">Jobs</h1>
          <p className="mt-2 max-w-xl text-sm text-[var(--muted)]">
            Live queue and completed analyses from the local FastAPI service.
          </p>
        </div>
        <Link className="btn btn-primary" to="/studio">
          Open Studio
        </Link>
      </header>

      <div className="mb-5 flex flex-wrap gap-1.5" role="tablist" aria-label="Job filters">
        {filters.map((f) => (
          <button
            key={f.id}
            type="button"
            role="tab"
            aria-selected={filter === f.id}
            className={`min-h-10 rounded-full border px-3.5 py-1.5 text-xs font-medium transition ${
              filter === f.id
                ? "border-[var(--tracking)] bg-[color-mix(in_srgb,var(--tracking)_12%,transparent)] text-[var(--tracking)]"
                : "border-[var(--b1)] text-[var(--muted)] hover:border-[var(--b2)] hover:text-[var(--text)]"
            }`}
            onClick={() => setFilter(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading && <p className="text-[var(--muted)]">Loading jobs…</p>}
      {error && (
        <p className="text-[var(--danger)]" role="alert">
          Could not load jobs. Is the API running on 127.0.0.1:8000?
        </p>
      )}

      {!isLoading && !filtered.length && (
        <div className="panel p-10 text-center">
          <p className="display text-xl">No jobs match this filter</p>
          <p className="mt-2 text-sm text-[var(--muted)]">Start a new analysis in Studio.</p>
          <Link className="btn btn-primary mt-5 inline-flex" to="/studio">
            Open Studio
          </Link>
        </div>
      )}

      {/* Mobile cards */}
      <ul className="space-y-3 md:hidden">
        {filtered.map((j) => (
          <li key={j.job_id} className="panel p-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="mono truncate text-[11px] text-[var(--dim)]">{j.job_id}</p>
                <p className="mt-1 text-sm font-medium capitalize">
                  {j.tracker}
                  {j.runtime ? (
                    <span className="text-[var(--dim)]"> · {j.runtime}</span>
                  ) : null}
                </p>
              </div>
              <span className="status-pill" style={{ color: statusColor(j.status), borderColor: statusColor(j.status) }}>
                {j.status.replaceAll("_", " ")}
              </span>
            </div>
            <p className="mt-3 mono text-xs text-[var(--muted)]">
              {j.overall_percent != null ? `${j.overall_percent}%` : "—"}
              {j.frames_total != null
                ? ` · ${j.frames_completed ?? 0}/${j.frames_total} frames`
                : ""}
            </p>
            <div className="mt-3 flex gap-2">
              <Link className="btn flex-1" to={`/jobs/${j.job_id}`}>
                Open
              </Link>
              {["succeeded", "review_required", "partial"].includes(j.status) && (
                <a className="btn" href={api.downloadUrl(j.job_id)}>
                  ZIP
                </a>
              )}
            </div>
          </li>
        ))}
      </ul>

      <div className="panel hidden overflow-hidden md:block">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead className="bg-[var(--s2)] text-xs uppercase tracking-wider text-[var(--dim)]">
              <tr>
                <th className="px-4 py-3 font-medium">Job</th>
                <th className="px-4 py-3 font-medium">Created</th>
                <th className="px-4 py-3 font-medium">Tracker</th>
                <th className="px-4 py-3 font-medium">State</th>
                <th className="px-4 py-3 font-medium">Progress</th>
                <th className="px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((j) => (
                <tr
                  key={j.job_id}
                  className="border-t border-[var(--b1)] transition hover:bg-[color-mix(in_srgb,var(--s2)_55%,transparent)]"
                >
                  <td className="mono px-4 py-3 text-xs text-[var(--muted)]">{j.job_id}</td>
                  <td className="px-4 py-3 text-xs text-[var(--muted)]">
                    {j.created_at ? new Date(j.created_at).toLocaleString() : "—"}
                  </td>
                  <td className="px-4 py-3">
                    <span className="font-medium">{j.tracker}</span>
                    {j.runtime ? (
                      <span className="block text-[10px] text-[var(--dim)]">{j.runtime}</span>
                    ) : null}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className="status-pill"
                      style={{ color: statusColor(j.status), borderColor: statusColor(j.status) }}
                    >
                      {j.status.replaceAll("_", " ")}
                    </span>
                    {j.status === "queued" && j.queue_position != null && (
                      <span className="mt-1 block text-[10px] text-[var(--warning)]">
                        queue #{j.queue_position}
                      </span>
                    )}
                    {j.error_code && (
                      <span className="mt-1 block text-[10px] text-[var(--danger)]">{j.error_code}</span>
                    )}
                  </td>
                  <td className="mono px-4 py-3 text-xs">
                    {j.overall_percent != null ? `${j.overall_percent}%` : "—"}
                    {j.frames_total != null && (
                      <span className="text-[var(--dim)]">
                        {" "}
                        ({j.frames_completed ?? 0}/{j.frames_total})
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <Link className="font-medium text-[var(--tracking)] underline-offset-2 hover:underline" to={`/jobs/${j.job_id}`}>
                      Open
                    </Link>
                    {["succeeded", "review_required", "partial"].includes(j.status) && (
                      <>
                        {" · "}
                        <a
                          className="font-medium text-[var(--tracking)] underline-offset-2 hover:underline"
                          href={api.downloadUrl(j.job_id)}
                        >
                          Download
                        </a>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
