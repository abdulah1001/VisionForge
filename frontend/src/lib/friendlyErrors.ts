import { ApiError } from "@/api/client";

const MESSAGES: Record<string, string> = {
  UPLOAD_TOO_LARGE: "That video is too large. Try a shorter clip or a smaller file.",
  UPLOAD_EMPTY: "The file looks empty. Pick another video and try again.",
  UNSUPPORTED_MEDIA: "This file type isn’t supported. Use MP4, MOV, WebM, or AVI.",
  VIDEO_UNREADABLE: "We couldn’t open that video. Re-export it and try again.",
  VIDEO_NO_FRAMES: "This video has no playable frames.",
  PREVIEW_NOT_FOUND: "The upload session expired. Please upload the video again.",
  DETECTION_FAILED: "Object analysis failed. Pause on a clearer frame and try again.",
  INVALID_INITIAL_MASK: "That selection didn’t work. Pick another object or draw a box.",
  INVALID_BOX: "The selection box is invalid. Draw a larger box around the object.",
  INVALID_SPEC: "Something was wrong with the request. Try again.",
  QUEUE_FULL: "The system is busy. Wait a moment and try again.",
  NOT_ACCEPTING: "The system isn’t ready for new jobs yet. Check System status.",
  CANCELLED: "Removal was cancelled.",
  INPUT_MISSING: "The uploaded video couldn’t be found. Please start over.",
  JOB_NOT_COMPLETE: "The result isn’t ready yet.",
  NOT_FOUND: "We couldn’t find that job. It may have been cleaned up.",
  HTTP_ERROR: "The server didn’t respond as expected. Try again.",
  NETWORK: "Network error. Check that the local server is running.",
  STUDIO_ERROR: "Something went wrong in the studio. Try again.",
  TRACK_UNRELIABLE:
    "We couldn’t follow this object reliably. Try selecting it from a clearer frame.",
  GPU_OOM: "This video is too large for the available GPU. Retry Optimized (~640p) and try again.",
  INPAINT_INCOMPLETE:
    "Object removal could not be completed. Your original video was not changed.",
  PIPELINE_FAILED: "Object removal could not be completed. Your original video was not changed.",
  NO_OBJECTS: "No objects found on this frame. Try another moment or select manually.",
};

function looksLikeOom(text: string): boolean {
  return /out of memory|outofmemory|cuda.*memory|gpu_oom|\boom\b/i.test(text);
}

export function friendlyErrorMessage(error: unknown): { title: string; message: string; code?: string } {
  if (error instanceof ApiError) {
    const raw = `${error.code} ${error.message}`;
    if (error.code === "GPU_OOM" || looksLikeOom(raw)) {
      return {
        title: "Unable to continue",
        message: MESSAGES.GPU_OOM,
        code: "GPU_OOM",
      };
    }
    return {
      title: "Unable to continue",
      message: MESSAGES[error.code] ?? error.message,
      code: error.code,
    };
  }
  if (error instanceof Error) {
    if (looksLikeOom(error.message)) {
      return { title: "Unable to continue", message: MESSAGES.GPU_OOM, code: "GPU_OOM" };
    }
    return { title: "Unable to continue", message: error.message };
  }
  return { title: "Unable to continue", message: "An unexpected error occurred." };
}
