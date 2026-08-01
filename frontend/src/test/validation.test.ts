import { describe, expect, it } from "vitest";
import {
  computeContainTransform,
  viewToNatural,
  naturalToView,
  validNaturalBox,
} from "@/lib/coords";
import { isTerminal, normalizeLabel, trackerAvailable, validBox } from "@/api/types";

describe("coordinate mapping", () => {
  it("round-trips letterboxed coordinates", () => {
    const t = computeContainTransform(1080, 1920, 800, 600);
    const [vx, vy] = naturalToView(100, 200, t);
    const [nx, ny] = viewToNatural(vx, vy, t);
    expect(nx).toBeCloseTo(100, 5);
    expect(ny).toBeCloseTo(200, 5);
  });

  it("rejects fixture-sized empty selection on portrait", () => {
    expect(validNaturalBox(null)).toBe(false);
    expect(validBox([20, 60, 60, 100], 1080, 1920)).toBe(true);
    expect(validNaturalBox([20, 60, 21, 61], 1080, 1920)).toBe(false);
  });
});

describe("status and labels", () => {
  it("treats review_required as terminal", () => {
    expect(isTerminal("review_required")).toBe(true);
    expect(isTerminal("partial")).toBe(true);
    expect(isTerminal("running")).toBe(false);
  });

  it("normalizes labels", () => {
    expect(normalizeLabel("  a  ")).toBe("a");
    expect(normalizeLabel("")).toBeNull();
  });

  it("does not invent native SAM availability", () => {
    expect(trackerAvailable("AVAILABLE_WSL2")).toBe(true);
    expect(trackerAvailable("UNAVAILABLE")).toBe(false);
  });
});
