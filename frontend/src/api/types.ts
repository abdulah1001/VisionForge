/** Typed VisionForge API contracts (match FastAPI responses). */
import { z } from "zod";

export const TrackerSchema = z.enum(["edgetam", "sam31"]);
export type Tracker = z.infer<typeof TrackerSchema>;

export const JobStatusSchema = z.enum([
  "queued",
  "running",
  "cancelling",
  "succeeded",
  "review_required",
  "partial",
  "failed",
  "cancelled",
]);
export type JobStatus = z.infer<typeof JobStatusSchema>;

export type Box = [number, number, number, number];

export interface JobSpec {
  tracker: Tracker;
  box: Box;
  labels: string[];
  max_frames?: number | null;
  processing_width?: number | null;
  processing_height?: number | null;
  analysis_mode?: "full" | "sampled";
  selection_mode?: "manual" | "automatic";
  mask_confirmed: boolean;
  operation?: "track_analyze" | "remove_object";
  selected_label?: string | null;
  anchor_time_sec?: number | null;
  quality_mode?: "standard" | "high";
}

export interface JobAccepted {
  job_id: string;
  status: JobStatus;
  tracker: string;
  status_url: string;
}

export interface JobStatusView {
  job_id: string;
  tracker: string;
  status: JobStatus;
  stage: string | null;
  stage_progress: { completed?: number | null; total?: number | null } | null;
  overall_percent: number | null;
  frames_completed: number | null;
  frames_total: number | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  queue_position: number | null;
  pipeline_run_id: string | null;
  warning_code: string | null;
  error_code: string | null;
  error_message: string | null;
  result_url: string | null;
  artifacts_url: string | null;
  download_url: string | null;
  runtime: string | null;
}

export interface ArtifactItem {
  id: string;
  group: string;
  path: string;
  size_bytes: number;
  name: string;
  preview_url: string | null;
}

export interface JobResult {
  job_id: string;
  status?: JobStatus | string;
  pipeline_run_id?: string;
  selected_tracker?: string;
  tracker_runtime?: string;
  wsl_distro?: string | null;
  frames?: {
    processed?: number;
    successful?: number;
    failed_count?: number;
    valid_masks?: number;
    invalid_masks?: number;
    empty_masks?: number;
  };
  artifact_counts?: {
    masks?: number;
    valid_masks?: number;
    invalid_masks?: number;
    empty_masks?: number;
    mask_files?: number;
    overlays?: number;
    crops?: number;
  };
  quality?: {
    recommended_status?: string;
    suspected_drift_count?: number;
    longest_failure_sequence?: number;
    frames_requiring_review?: number[];
    valid_ratio?: number;
    note?: string;
    warnings?: string[];
  };
  annotated_video?: {
    available?: boolean;
    width?: number;
    height?: number;
    fps?: number;
    frames?: number;
    duration_sec?: number;
    codec?: string;
    pixel_format?: string;
    faststart?: boolean;
    audio_note?: string;
    audio?: {
      audio_present?: boolean;
      audio_preserved?: boolean;
      audio_present_in_source?: boolean;
      audio_codec?: string | null;
      warning?: string | null;
      note?: string;
      ffprobe_confirmed?: boolean;
    };
    artifact_name?: string;
  };
  dinov3_feature_shape?: number[];
  identity_summary?: Record<string, unknown>;
  mobileclip2?: {
    image_feature_shape?: number[];
    text_feature_shape?: number[];
    mean_scores?: Record<string, number>;
    highest_scoring_aggregate_label?: string;
    valid_crops_used?: number;
    skipped?: boolean;
    note?: string;
  };
  stages_timing?: Record<string, Record<string, number | null | undefined>>;
  total_duration_sec?: number;
  warnings?: string[];
  real_cuda_inference?: boolean;
  offline_local_only?: boolean;
  mock_or_fallback_used?: boolean;
  download_url?: string;
}

export interface MediaProbe {
  preview_id: string;
  filename: string;
  kind: string;
  mime_hint?: string;
  width: number;
  height: number;
  duration_sec: number | null;
  fps: number | null;
  estimated_frames: number | null;
  orientation_normalized?: boolean;
  first_frame_url: string;
  limits?: Record<string, number>;
  detector?: {
    status: string;
    detail?: string;
    name?: string;
    supports_text_prompt?: boolean;
    supports_class_agnostic?: boolean;
  };
}

export interface Capabilities {
  trackers: {
    edgetam: { status: string; runtime?: string; detail?: string };
    sam31: {
      status: string;
      runtime?: string;
      wsl_distro?: string;
      detail?: string;
      available_native_windows?: boolean;
    };
  };
  detector?: {
    status: string;
    name?: string;
    detail?: string;
    supports_text_prompt?: boolean;
    supports_class_agnostic?: boolean;
  };
  models?: Record<string, { status: string; checkpoint?: string; detail?: string }>;
  cuda?: { available?: boolean; name?: string | null };
  wsl2?: { distro?: string; sam31_status?: string; required_for_sam31?: boolean };
  notes?: string[];
}

export interface ReadyState {
  status: string;
  worker?: {
    alive?: boolean;
    active_job_id?: string | null;
    gpu_concurrency?: number;
  };
  queue?: { queued?: number; max_queued?: number; accepting?: boolean };
  artifacts?: {
    jobs_root_writable?: boolean;
    free_bytes?: number;
    min_free_bytes?: number;
  };
  accepting_jobs?: boolean;
}

export const JobAcceptedSchema = z.object({
  job_id: z.string(),
  status: JobStatusSchema,
  tracker: z.string(),
  status_url: z.string(),
});

export const JobStatusViewSchema = z.object({
  job_id: z.string(),
  tracker: z.string(),
  status: JobStatusSchema,
  stage: z.string().nullable().optional(),
  stage_progress: z
    .object({
      completed: z.number().nullable().optional(),
      total: z.number().nullable().optional(),
    })
    .nullable()
    .optional(),
  overall_percent: z.number().nullable().optional(),
  frames_completed: z.number().nullable().optional(),
  frames_total: z.number().nullable().optional(),
  created_at: z.string().nullable().optional(),
  started_at: z.string().nullable().optional(),
  finished_at: z.string().nullable().optional(),
  queue_position: z.number().nullable().optional(),
  pipeline_run_id: z.string().nullable().optional(),
  warning_code: z.string().nullable().optional(),
  error_code: z.string().nullable().optional(),
  error_message: z.string().nullable().optional(),
  result_url: z.string().nullable().optional(),
  artifacts_url: z.string().nullable().optional(),
  download_url: z.string().nullable().optional(),
  runtime: z.string().nullable().optional(),
});

export const TERMINAL = new Set<JobStatus>([
  "succeeded",
  "review_required",
  "partial",
  "failed",
  "cancelled",
]);

export function isTerminal(status: JobStatus): boolean {
  return TERMINAL.has(status);
}

export function trackerAvailable(status: string | undefined): boolean {
  return status === "AVAILABLE" || status === "AVAILABLE_WSL2";
}

export function validBox([x1, y1, x2, y2]: Box, frameW?: number, frameH?: number): boolean {
  if (![x1, y1, x2, y2].every(Number.isFinite)) return false;
  if (x2 <= x1 || y2 <= y1) return false;
  if (x2 - x1 < 2 || y2 - y1 < 2) return false;
  if (frameW != null && frameH != null) {
    if (x2 <= 0 || y2 <= 0 || x1 >= frameW || y1 >= frameH) return false;
  }
  return true;
}

export function normalizeLabel(raw: string): string | null {
  const s = raw.trim();
  if (!s) return null;
  if ([...s].some((ch) => ch.charCodeAt(0) < 32)) return null;
  return s;
}

export const STAGES = [
  "validating",
  "preparing_input",
  "tracking",
  "creating_artifacts",
  "validating_masks",
  "encoding_video",
  "dinov3",
  "mobileclip2",
  "recovering_identity",
  "finalizing",
  "completed",
] as const;
