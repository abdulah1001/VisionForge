import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { StudioPage } from "@/pages/StudioPage";
import { useRemoverStore } from "@/store/removerStore";

vi.mock("@/api/hooks", () => ({
  useCapabilities: () => ({
    data: {
      trackers: {
        edgetam: { status: "AVAILABLE" },
        sam31: { status: "UNAVAILABLE" },
      },
    },
  }),
  useCreateJob: () => ({ mutate: vi.fn(), isPending: false, error: null }),
  useCancelJob: () => ({ mutate: vi.fn(), isPending: false, error: null }),
  useJob: () => ({ data: undefined }),
}));

function renderStudio() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <StudioPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Studio Remove Object button", () => {
  beforeEach(() => {
    localStorage.removeItem("visionforge-active-job");
    useRemoverStore.getState().reset();
  });

  it("keeps Remove Object disabled without a selection", () => {
    const url = "blob:mock-video";
    useRemoverStore.getState().setUpload(
      new File([new Uint8Array([0, 0, 0])], "clip.mp4", { type: "video/mp4" }),
      url,
      {
        filename: "clip.mp4",
        width: 640,
        height: 360,
        durationSec: 2,
        fps: 24,
        sizeBytes: 1024,
        estimatedFrames: 48,
      },
      "preview-1",
    );
    renderStudio();
    const btn = screen.getByTestId("remove-object-btn");
    expect(btn).toBeDisabled();
  });

  it("enables Remove Object after a valid selection", () => {
    const url = "blob:mock-video";
    useRemoverStore.getState().setUpload(
      new File([new Uint8Array([0, 0, 0])], "clip.mp4", { type: "video/mp4" }),
      url,
      {
        filename: "clip.mp4",
        width: 640,
        height: 360,
        durationSec: 2,
        fps: 24,
        sizeBytes: 1024,
        estimatedFrames: 48,
      },
      "preview-1",
    );
    useRemoverStore.getState().setManualBox([20, 20, 120, 120]);
    renderStudio();
    expect(screen.getByTestId("remove-object-btn")).toBeEnabled();
  });
});
