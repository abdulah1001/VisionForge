/** Source-pixel ↔ display-rect conversion for object-fit: contain video overlays. */

export type BoxXyxy = [number, number, number, number];

export interface ContainRect {
  offsetX: number;
  offsetY: number;
  displayWidth: number;
  displayHeight: number;
  scale: number;
  naturalWidth: number;
  naturalHeight: number;
  containerWidth: number;
  containerHeight: number;
}

/** Letterboxed content rect for object-fit: contain. */
export function getObjectFitContainRect(
  naturalWidth: number,
  naturalHeight: number,
  containerWidth: number,
  containerHeight: number,
): ContainRect {
  const nw = Math.max(1, naturalWidth);
  const nh = Math.max(1, naturalHeight);
  const cw = Math.max(1, containerWidth);
  const ch = Math.max(1, containerHeight);
  const scale = Math.min(cw / nw, ch / nh);
  const displayWidth = nw * scale;
  const displayHeight = nh * scale;
  return {
    offsetX: (cw - displayWidth) / 2,
    offsetY: (ch - displayHeight) / 2,
    displayWidth,
    displayHeight,
    scale,
    naturalWidth: nw,
    naturalHeight: nh,
    containerWidth: cw,
    containerHeight: ch,
  };
}

/** Source XYXY → CSS pixel box relative to the container. */
export function sourceBoxToDisplayBox(box: BoxXyxy, rect: ContainRect): BoxXyxy {
  const [x1, y1, x2, y2] = box;
  return [
    rect.offsetX + x1 * rect.scale,
    rect.offsetY + y1 * rect.scale,
    rect.offsetX + x2 * rect.scale,
    rect.offsetY + y2 * rect.scale,
  ];
}

/** Container-relative CSS XYXY → source pixels. */
export function displayBoxToSourceBox(box: BoxXyxy, rect: ContainRect): BoxXyxy {
  const inv = rect.scale > 0 ? 1 / rect.scale : 0;
  const [x1, y1, x2, y2] = box;
  return [
    (x1 - rect.offsetX) * inv,
    (y1 - rect.offsetY) * inv,
    (x2 - rect.offsetX) * inv,
    (y2 - rect.offsetY) * inv,
  ];
}

/** Client (viewport) point → source pixels. */
export function clientPointToSource(
  clientX: number,
  clientY: number,
  container: DOMRect,
  rect: ContainRect,
): [number, number] {
  const lx = clientX - container.left;
  const ly = clientY - container.top;
  const inv = rect.scale > 0 ? 1 / rect.scale : 0;
  return [(lx - rect.offsetX) * inv, (ly - rect.offsetY) * inv];
}

export function clampSourceBox(box: BoxXyxy, w: number, h: number): BoxXyxy {
  let [x1, y1, x2, y2] = box;
  x1 = Math.max(0, Math.min(w, x1));
  y1 = Math.max(0, Math.min(h, y1));
  x2 = Math.max(0, Math.min(w, x2));
  y2 = Math.max(0, Math.min(h, y2));
  if (x2 < x1) [x1, x2] = [x2, x1];
  if (y2 < y1) [y1, y2] = [y2, y1];
  return [x1, y1, x2, y2];
}

export function pointInBox(x: number, y: number, box: BoxXyxy): boolean {
  const [x1, y1, x2, y2] = box;
  return x >= x1 && x <= x2 && y >= y1 && y <= y2;
}

/** Title-case a detector label for display. */
export function titleCaseLabel(raw: string | null | undefined): string {
  const s = (raw ?? "").trim();
  if (!s) return "Object";
  return s
    .split(/[\s_]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

/** Map pipeline stages to user-friendly copy. */
export function friendlyProcessingStage(stage: string | null | undefined): string {
  const s = (stage ?? "").toLowerCase().trim();
  if (!s || s === "queued") return "Preparing video";
  if (
    s === "preparing" ||
    s === "preparing_input" ||
    s === "validating" ||
    s === "creating_artifacts" ||
    s === "validating_masks"
  ) {
    return "Preparing video";
  }
  if (s === "tracking" || s === "recovering_identity") return "Following selected object";
  if (s === "inpainting" || s === "dinov3" || s === "mobileclip2") {
    return "Rebuilding background";
  }
  if (
    s === "encoding" ||
    s === "encoding_video" ||
    s === "finalizing" ||
    s === "completed"
  ) {
    return "Finalizing result";
  }
  if (s === "cancelling" || s === "cancelled") return "Cancelling…";
  if (s === "failed") return "Something went wrong";
  return "Working on your video";
}
