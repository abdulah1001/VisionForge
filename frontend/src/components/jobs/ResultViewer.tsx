import type { JobResult } from "@/api/types";

export function ResultViewer({ result }: { result: JobResult }) {
  return (
    <section className="panel p-4">
      <h3 className="display text-lg">Result summary</h3>
      <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Selected tracker" value={result.selected_tracker ?? "—"} />
        <Field label="Runtime" value={result.tracker_runtime ?? "—"} />
        <Field label="WSL distro" value={result.wsl_distro ?? "—"} />
        <Field
          label="Frames successful"
          value={`${result.frames?.successful ?? "—"} / ${result.frames?.processed ?? "—"}`}
        />
        <Field
          label="Failed frames"
          value={String(result.frames?.failed_count ?? "—")}
        />
        <Field
          label="Duration"
          value={
            result.total_duration_sec != null
              ? `${result.total_duration_sec.toFixed(1)}s`
              : "—"
          }
        />
        <Field
          label="Real CUDA"
          value={result.real_cuda_inference ? "Yes" : result.real_cuda_inference === false ? "No" : "—"}
        />
        <Field
          label="Offline local"
          value={result.offline_local_only ? "Yes" : "—"}
        />
        <Field
          label="Mock / fallback"
          value={result.mock_or_fallback_used ? "USED" : "None"}
        />
      </div>
      {result.warnings?.length ? (
        <ul className="mt-4 text-xs text-[var(--warning)]">
          {result.warnings.map((w, i) => (
            <li key={`${i}-${w}`}>{w}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-[var(--dim)]">{label}</p>
      <p className="mono mt-0.5">{value}</p>
    </div>
  );
}
