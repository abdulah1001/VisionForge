/** Extract first frame + metadata from a local video/image File. */

export interface MediaMeta {
  filename: string;
  kind: "video" | "image" | "zip";
  width: number;
  height: number;
  durationSec: number | null;
  fps: number | null;
  estimatedFrames: number | null;
  previewUrl: string;
}

function revokeLater(url: string) {
  // caller owns lifecycle; helper for clarity
  return url;
}

export async function extractFirstFrameFromVideo(file: File): Promise<MediaMeta> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    const onErr = () => reject(new Error("Unable to decode video in browser"));
    video.addEventListener("loadeddata", () => resolve(), { once: true });
    video.addEventListener("error", onErr, { once: true });
  });

  // Seek to first decodable frame
  try {
    video.currentTime = 0;
    await new Promise<void>((resolve) => {
      video.addEventListener("seeked", () => resolve(), { once: true });
      window.setTimeout(() => resolve(), 500);
    });
  } catch {
    /* ignore */
  }

  const width = video.videoWidth || 0;
  const height = video.videoHeight || 0;
  if (!width || !height) {
    URL.revokeObjectURL(url);
    throw new Error("Video has no displayable dimensions");
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    URL.revokeObjectURL(url);
    throw new Error("Canvas unavailable");
  }
  ctx.drawImage(video, 0, 0, width, height);
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob((b) => resolve(b), "image/jpeg", 0.92),
  );
  URL.revokeObjectURL(url);
  if (!blob) throw new Error("Failed to capture first frame");

  const previewUrl = URL.createObjectURL(blob);
  const duration = Number.isFinite(video.duration) ? video.duration : null;
  const fps = null;
  const estimatedFrames =
    duration != null && duration > 0 ? Math.max(1, Math.round(duration * 24)) : null;

  return {
    filename: file.name,
    kind: "video",
    width,
    height,
    durationSec: duration,
    fps,
    estimatedFrames,
    previewUrl: revokeLater(previewUrl),
  };
}

export async function extractImageMeta(file: File): Promise<MediaMeta> {
  const url = URL.createObjectURL(file);
  const size = await new Promise<{ w: number; h: number }>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => reject(new Error("Unable to load image"));
    img.src = url;
  });
  return {
    filename: file.name,
    kind: "image",
    width: size.w,
    height: size.h,
    durationSec: null,
    fps: null,
    estimatedFrames: 1,
    previewUrl: url,
  };
}
