import {
  JobAcceptedSchema,
  JobStatusViewSchema,
  normalizeLabel,
  trackerAvailable,
  validBox,
} from "@/api/types";

describe("domain contracts", () => {
  it("parses job accepted", () => {
    expect(
      JobAcceptedSchema.parse({
        job_id: "j1",
        status: "queued",
        tracker: "edgetam",
        status_url: "/v1/jobs/j1",
      }).job_id,
    ).toBe("j1");
  });

  it("parses job status", () => {
    const j = JobStatusViewSchema.parse({
      job_id: "j1",
      tracker: "sam31",
      status: "running",
      overall_percent: 40,
    });
    expect(j.job_id).toBe("j1");
  });

  it("validates boxes", () => {
    expect(validBox([0, 0, 10, 10])).toBe(true);
    expect(validBox([10, 0, 2, 10])).toBe(false);
    expect(validBox([20, 60, 60, 100], 256, 256)).toBe(true);
    expect(validBox([300, 0, 310, 10], 256, 256)).toBe(false);
  });

  it("tracks availability without silent fallback", () => {
    expect(trackerAvailable("AVAILABLE")).toBe(true);
    expect(trackerAvailable("AVAILABLE_WSL2")).toBe(true);
    expect(trackerAvailable("UNAVAILABLE")).toBe(false);
    expect(trackerAvailable("AVAILABLE_NATIVE_WINDOWS")).toBe(false);
  });

  it("normalizes labels", () => {
    expect(normalizeLabel("  a red circle  ")).toBe("a red circle");
    expect(normalizeLabel("")).toBeNull();
    expect(normalizeLabel("bad\nlabel")).toBeNull();
  });
});
