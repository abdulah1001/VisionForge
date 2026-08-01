import { create } from "zustand";
import type { Box, Tracker } from "@/api/types";

export type RemoverPhase = "upload" | "edit" | "processing" | "result" | "failed";
export type QualityMode = "standard" | "high";
export type SelectionMode = "automatic" | "manual";

export interface DetectedObject {
  candidate_id: string;
  box_xyxy: Box;
  score: number | null;
  label: string | null;
}

export interface VideoMeta {
  filename: string;
  width: number;
  height: number;
  durationSec: number | null;
  fps: number | null;
  sizeBytes: number;
  estimatedFrames: number | null;
}

interface RemoverState {
  phase: RemoverPhase;
  file: File | null;
  videoUrl: string | null;
  previewId: string | null;
  meta: VideoMeta | null;
  candidates: DetectedObject[];
  selectedId: string | null;
  box: Box | null;
  selectedLabel: string | null;
  selectionMode: SelectionMode;
  manualDraw: boolean;
  qualityMode: QualityMode;
  anchorTimeSec: number;
  activeJobId: string | null;
  setPhase: (phase: RemoverPhase) => void;
  setUpload: (file: File, videoUrl: string, meta: VideoMeta, previewId: string) => void;
  setCandidates: (c: DetectedObject[]) => void;
  selectObject: (id: string | null, box?: Box | null, label?: string | null) => void;
  setManualBox: (box: Box) => void;
  setManualDraw: (v: boolean) => void;
  clearSelection: () => void;
  setQualityMode: (m: QualityMode) => void;
  setAnchorTimeSec: (t: number) => void;
  setActiveJob: (id: string | null) => void;
  backToDetect: () => void;
  reset: () => void;
}

function revoke(url: string | null) {
  if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
}

const initial = {
  phase: "upload" as RemoverPhase,
  file: null as File | null,
  videoUrl: null as string | null,
  previewId: null as string | null,
  meta: null as VideoMeta | null,
  candidates: [] as DetectedObject[],
  selectedId: null as string | null,
  box: null as Box | null,
  selectedLabel: null as string | null,
  selectionMode: "automatic" as SelectionMode,
  manualDraw: false,
  qualityMode: "standard" as QualityMode,
  anchorTimeSec: 0,
  activeJobId: null as string | null,
};

export function pickTracker(caps: {
  trackers: { edgetam: { status: string }; sam31: { status: string } };
} | undefined): Tracker {
  const edge = caps?.trackers.edgetam.status;
  const sam = caps?.trackers.sam31.status;
  if (edge === "AVAILABLE" || edge === "AVAILABLE_WSL2") return "edgetam";
  if (sam === "AVAILABLE" || sam === "AVAILABLE_WSL2") return "sam31";
  return "edgetam";
}

export const useRemoverStore = create<RemoverState>((set, get) => ({
  ...initial,
  setPhase: (phase) =>
    set((state) => (state.phase === phase ? state : { phase })),
  setUpload: (file, videoUrl, meta, previewId) => {
    revoke(get().videoUrl);
    set({
      ...initial,
      phase: "edit",
      file,
      videoUrl,
      meta,
      previewId,
    });
  },
  setCandidates: (candidates) =>
    set({
      candidates,
      selectedId: null,
      box: null,
      selectedLabel: null,
      selectionMode: "automatic",
      manualDraw: false,
    }),
  selectObject: (id, box, label) =>
    set({
      selectedId: id,
      box: box ?? get().box,
      selectedLabel: label ?? null,
      selectionMode: id ? "automatic" : get().selectionMode,
      manualDraw: false,
    }),
  setManualBox: (box) =>
    set({
      box,
      selectedId: null,
      selectedLabel: "Selected region",
      selectionMode: "manual",
      manualDraw: false,
    }),
  setManualDraw: (manualDraw) => set({ manualDraw }),
  clearSelection: () =>
    set({
      selectedId: null,
      box: null,
      selectedLabel: null,
      selectionMode: "automatic",
      manualDraw: false,
    }),
  setQualityMode: (qualityMode) => set({ qualityMode }),
  setAnchorTimeSec: (anchorTimeSec) => set({ anchorTimeSec }),
  setActiveJob: (activeJobId) =>
    set((state) => (state.activeJobId === activeJobId ? state : { activeJobId })),
  backToDetect: () =>
    set({
      phase: "edit",
      activeJobId: null,
      candidates: [],
      selectedId: null,
      box: null,
      selectedLabel: null,
      selectionMode: "automatic",
      manualDraw: false,
    }),
  reset: () => {
    revoke(get().videoUrl);
    set({ ...initial });
  },
}));
