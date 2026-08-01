/** Fit media into a max long-side for 8GB-class GPUs (even dims for encoders). */
export function fitProcessingSize(
  width: number,
  height: number,
  maxSide = 720,
): { width: number; height: number } {
  const w = Math.max(1, Math.round(width));
  const h = Math.max(1, Math.round(height));
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const even = (n: number) => Math.max(2, Math.round(n / 2) * 2);
  return { width: even(w * scale), height: even(h * scale) };
}
