import { useCapabilities, useReady } from "@/api/hooks";
import { trackerAvailable } from "@/api/types";

export function SystemPage() {
  const caps = useCapabilities();
  const ready = useReady();

  const c = caps.data;
  const r = ready.data;

  return (
    <div className="mx-auto max-w-[1000px] space-y-6 px-4 py-8 lg:px-8">
      <header>
        <p className="eyebrow">Runtime</p>
        <h1 className="display mt-1 text-4xl">System</h1>
        <p className="mt-2 max-w-xl text-sm text-[var(--muted)]">
          Live capability and readiness from the local VisionForge API. Paths, secrets, and
          environment values are never shown here.
        </p>
      </header>

      <section className="panel grid gap-4 p-5 sm:grid-cols-2">
        <Row label="API" value={ready.isError ? "Unreachable" : r?.status ?? "Checking"} />
        <Row
          label="Accepting jobs"
          value={r?.accepting_jobs === false ? "No" : r?.accepting_jobs ? "Yes" : "—"}
        />
        <Row
          label="Worker alive"
          value={r?.worker?.alive == null ? "—" : r.worker.alive ? "Yes" : "No"}
        />
        <Row
          label="Maximum active GPU jobs"
          value={String(r?.worker?.gpu_concurrency ?? 1)}
        />
        <Row label="Queued" value={String(r?.queue?.queued ?? "—")} />
        <Row
          label="Active job"
          value={r?.worker?.active_job_id ?? "None"}
          mono
        />
        <Row
          label="CUDA"
          value={
            c?.cuda?.available
              ? `Available${c.cuda.name ? ` · ${c.cuda.name}` : ""}`
              : "Unavailable"
          }
        />
        <Row
          label="Artifact storage"
          value={
            r?.artifacts?.jobs_root_writable == null
              ? "—"
              : r.artifacts.jobs_root_writable
                ? "Writable"
                : "Not writable"
          }
        />
      </section>

      <section className="panel p-5">
        <h2 className="display text-xl">Trackers</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <TrackerCard
            name="EdgeTAM"
            status={c?.trackers.edgetam.status}
            runtime="Native Windows CUDA"
            detail={c?.trackers.edgetam.detail}
          />
          <TrackerCard
            name="SAM 3.1"
            status={c?.trackers.sam31.status}
            runtime={`WSL2 CUDA · ${c?.trackers.sam31.wsl_distro ?? c?.wsl2?.distro ?? "—"}`}
            detail={
              c?.trackers.sam31.available_native_windows === false
                ? `${c?.trackers.sam31.detail ?? ""} Never available as native Windows.`
                : c?.trackers.sam31.detail
            }
          />
        </div>
      </section>

      <section className="panel p-5">
        <h2 className="display text-xl">Models</h2>
        <ul className="mt-3 space-y-2">
          {c?.models &&
            Object.entries(c.models).map(([id, m]) => (
              <li
                key={id}
                className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--b1)] py-2 text-sm last:border-0"
              >
                <span className="mono">{id}</span>
                <span
                  style={{
                    color: trackerAvailable(m.status) || m.status === "AVAILABLE"
                      ? "var(--success)"
                      : "var(--danger)",
                  }}
                >
                  {m.status}
                </span>
              </li>
            ))}
          {!c?.models && <li className="text-[var(--muted)]">Loading…</li>}
        </ul>
      </section>

      {c?.notes?.length ? (
        <ul className="text-xs text-[var(--dim)]">
          {c.notes.map((n) => (
            <li key={n}>• {n}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Row({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <p className="text-xs text-[var(--dim)]">{label}</p>
      <p className={`mt-1 text-sm ${mono ? "mono text-xs" : ""}`}>{value}</p>
    </div>
  );
}

function TrackerCard({
  name,
  status,
  runtime,
  detail,
}: {
  name: string;
  status?: string;
  runtime: string;
  detail?: string;
}) {
  const ok = trackerAvailable(status);
  return (
    <div className="rounded-xl border border-[var(--b1)] bg-[var(--s2)] p-4">
      <div className="flex items-center justify-between">
        <strong>{name}</strong>
        <span style={{ color: ok ? "var(--success)" : "var(--danger)" }}>
          {status ?? "Checking"}
        </span>
      </div>
      <p className="mt-2 text-xs text-[var(--muted)]">{runtime}</p>
      {detail && <p className="mt-2 text-xs text-[var(--dim)]">{detail}</p>}
    </div>
  );
}
