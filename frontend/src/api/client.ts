import {
  JobAcceptedSchema,
  JobStatusViewSchema,
  type ArtifactItem,
  type Capabilities,
  type JobAccepted,
  type JobResult,
  type JobSpec,
  type JobStatusView,
  type MediaProbe,
  type ReadyState,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  code: string;
  status: number;
  jobId?: string;

  constructor(code: string, message: string, status: number, jobId?: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.jobId = jobId;
  }
}

async function parseBody(response: Response): Promise<unknown> {
  if (response.status === 204) return null;
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { error: { code: "BAD_JSON", message: text.slice(0, 200) } };
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  const body = await parseBody(response);
  if (!response.ok) {
    const err = body as { error?: { code?: string; message?: string; job_id?: string } };
    throw new ApiError(
      err.error?.code ?? "HTTP_ERROR",
      err.error?.message ?? `Request failed (${response.status})`,
      response.status,
      err.error?.job_id,
    );
  }
  return body as T;
}

function normalizeJob(raw: unknown): JobStatusView {
  const parsed = JobStatusViewSchema.parse(raw);
  return {
    job_id: parsed.job_id,
    tracker: parsed.tracker,
    status: parsed.status,
    stage: parsed.stage ?? null,
    stage_progress: parsed.stage_progress ?? null,
    overall_percent: parsed.overall_percent ?? null,
    frames_completed: parsed.frames_completed ?? null,
    frames_total: parsed.frames_total ?? null,
    created_at: parsed.created_at ?? null,
    started_at: parsed.started_at ?? null,
    finished_at: parsed.finished_at ?? null,
    queue_position: parsed.queue_position ?? null,
    pipeline_run_id: parsed.pipeline_run_id ?? null,
    warning_code: parsed.warning_code ?? null,
    error_code: parsed.error_code ?? null,
    error_message: parsed.error_message ?? null,
    result_url: parsed.result_url ?? null,
    artifacts_url: parsed.artifacts_url ?? null,
    download_url: parsed.download_url ?? null,
    runtime: parsed.runtime ?? null,
  };
}

export const api = {
  health: () => request<Record<string, unknown>>("/health"),
  ready: () => request<ReadyState>("/ready"),
  capabilities: () => request<Capabilities>("/v1/capabilities"),
  jobs: async (): Promise<JobStatusView[]> => {
    const body = await request<{ jobs: unknown[] }>("/v1/jobs");
    return (body.jobs ?? []).map(normalizeJob);
  },
  job: async (id: string): Promise<JobStatusView> =>
    normalizeJob(await request(`/v1/jobs/${encodeURIComponent(id)}`)),
  createJob: async (file: File, spec: JobSpec): Promise<JobAccepted> => {
    const data = new FormData();
    data.append("input", file, file.name);
    data.append("spec", JSON.stringify(spec));
    const body = await request<unknown>("/v1/jobs", { method: "POST", body: data });
    return JobAcceptedSchema.parse(body);
  },
  cancelJob: async (id: string): Promise<{ job_id: string; status: string }> =>
    request(`/v1/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  result: (id: string) =>
    request<JobResult>(`/v1/jobs/${encodeURIComponent(id)}/result`),
  artifacts: async (id: string): Promise<ArtifactItem[]> => {
    const body = await request<{ artifacts: ArtifactItem[] }>(
      `/v1/jobs/${encodeURIComponent(id)}/artifacts`,
    );
    return body.artifacts ?? [];
  },
  artifactUrl: (jobId: string, artifactId: string) =>
    `${BASE}/v1/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(artifactId)}`,
  downloadUrl: (jobId: string) =>
    `${BASE}/v1/jobs/${encodeURIComponent(jobId)}/download`,
  /** Cleaned MP4 for remove_object jobs. */
  cleanedUrl: (jobId: string) =>
    `${BASE}/v1/jobs/${encodeURIComponent(jobId)}/cleaned`,
  cleanedDownloadUrl: (jobId: string) =>
    `${BASE}/v1/jobs/${encodeURIComponent(jobId)}/cleaned`,
  probeMedia: async (file: File): Promise<MediaProbe> => {
    const data = new FormData();
    data.append("input", file, file.name);
    return request<MediaProbe>("/v1/media/probe", { method: "POST", body: data });
  },
  /** Upload probe with real XHR upload progress (0–100). */
  probeMediaWithProgress: (
    file: File,
    onProgress?: (percent: number) => void,
  ): Promise<MediaProbe> =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const data = new FormData();
      data.append("input", file, file.name);
      xhr.open("POST", `${BASE}/v1/media/probe`);
      xhr.upload.onprogress = (ev) => {
        if (!onProgress) return;
        if (ev.lengthComputable && ev.total > 0) {
          onProgress(Math.min(99, Math.round((ev.loaded / ev.total) * 100)));
        } else {
          onProgress(Math.min(90, Math.round((ev.loaded / Math.max(file.size, 1)) * 100)));
        }
      };
      xhr.onload = () => {
        let body: unknown = null;
        try {
          body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        } catch {
          body = { error: { code: "BAD_JSON", message: xhr.responseText.slice(0, 200) } };
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          onProgress?.(100);
          resolve(body as MediaProbe);
          return;
        }
        const err = body as { error?: { code?: string; message?: string } };
        reject(
          new ApiError(
            err.error?.code ?? "HTTP_ERROR",
            err.error?.message ?? `Request failed (${xhr.status})`,
            xhr.status,
          ),
        );
      };
      xhr.onerror = () =>
        reject(new ApiError("NETWORK", "Network error during upload", 0));
      xhr.send(data);
    }),
  mediaFrameUrl: (previewId: string) =>
    `${BASE}/v1/media/${encodeURIComponent(previewId)}/frame`,
  maskPreview: async (
    previewId: string,
    spec: { box: number[]; tracker?: string; labels?: string[] },
  ) => {
    const data = new FormData();
    data.append("spec", JSON.stringify(spec));
    return request<{
      preview_id: string;
      diagnostics: Record<string, unknown>;
      mask_url: string;
      overlay_url: string;
      message: string;
    }>(`/v1/media/${encodeURIComponent(previewId)}/mask-preview`, {
      method: "POST",
      body: data,
    });
  },
  candidates: async (
    previewId: string,
    opts?: { textPrompt?: string; timeSec?: number },
  ) => {
    const data = new FormData();
    const textPrompt = opts?.textPrompt;
    if (textPrompt && textPrompt.trim()) {
      data.append("text_prompt", textPrompt.trim());
    }
    if (opts?.timeSec != null && Number.isFinite(opts.timeSec)) {
      data.append("time_sec", String(opts.timeSec));
    }
    return request<{
      status: string;
      candidates: Array<{
        candidate_id: string;
        box_xyxy: number[];
        score: number | null;
        label: string | null;
        has_mask: boolean;
      }>;
      detail?: string | null;
      detector?: Record<string, unknown>;
      text_prompt?: string | null;
    }>(`/v1/media/${encodeURIComponent(previewId)}/candidates`, {
      method: "POST",
      body: data,
    });
  },
  submitCorrection: async (
    jobId: string,
    body: {
      frame_index: number;
      box: number[];
      tracker: string;
      mask_confirmed: boolean;
      labels?: string[];
      analysis_mode?: string;
      max_frames?: number | null;
    },
  ) =>
    request<{
      job_id: string;
      status: string;
      tracker: string;
      status_url: string;
      parent_job_id?: string;
      revision_id?: string;
    }>(`/v1/jobs/${encodeURIComponent(jobId)}/corrections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
