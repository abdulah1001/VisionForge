/** Coordinate conversion between display canvas and natural source pixels. */

export type Box = [number, number, number, number];

export interface ViewTransform {
  naturalWidth: number;
  naturalHeight: number;
  viewWidth: number;
  viewHeight: number;
  offsetX: number;
  offsetY: number;
  scale: number;
}

/** object-fit: contain letterbox mapping */
export function computeContainTransform(
  naturalWidth: number,
  naturalHeight: number,
  viewWidth: number,
  viewHeight: number,
): ViewTransform {
  const nw = Math.max(1, naturalWidth);
  const nh = Math.max(1, naturalHeight);
  const vw = Math.max(1, viewWidth);
  const vh = Math.max(1, viewHeight);
  const scale = Math.min(vw / nw, vh / nh);
  const drawW = nw * scale;
  const drawH = nh * scale;
  return {
    naturalWidth: nw,
    naturalHeight: nh,
    viewWidth: vw,
    viewHeight: vh,
    offsetX: (vw - drawW) / 2,
    offsetY: (vh - drawH) / 2,
    scale,
  };
}

export function viewToNatural(
  x: number,
  y: number,
  t: ViewTransform,
): [number, number] {
  return [(x - t.offsetX) / t.scale, (y - t.offsetY) / t.scale];
}

export function naturalToView(
  x: number,
  y: number,
  t: ViewTransform,
): [number, number] {
  return [x * t.scale + t.offsetX, y * t.scale + t.offsetY];
}

export function clampBox(box: Box, w: number, h: number): Box {
  let [x1, y1, x2, y2] = box;
  x1 = Math.max(0, Math.min(w, x1));
  y1 = Math.max(0, Math.min(h, y1));
  x2 = Math.max(0, Math.min(w, x2));
  y2 = Math.max(0, Math.min(h, y2));
  if (x2 < x1) [x1, x2] = [x2, x1];
  if (y2 < y1) [y1, y2] = [y2, y1];
  return [x1, y1, x2, y2];
}

export function validNaturalBox(
  box: Box | null,
  frameW?: number,
  frameH?: number,
): boolean {
  if (!box) return false;
  const [x1, y1, x2, y2] = box;
  if (![x1, y1, x2, y2].every(Number.isFinite)) return false;
  if (x2 <= x1 || y2 <= y1) return false;
  if (x2 - x1 < 2 || y2 - y1 < 2) return false;
  if (frameW != null && frameH != null) {
    if (x2 <= 0 || y2 <= 0 || x1 >= frameW || y1 >= frameH) return false;
  }
  return true;
}
