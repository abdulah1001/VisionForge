import { describe, expect, it } from "vitest";
import {
  clampSourceBox,
  clientPointToSource,
  displayBoxToSourceBox,
  friendlyProcessingStage,
  getObjectFitContainRect,
  pointInBox,
  sourceBoxToDisplayBox,
  titleCaseLabel,
} from "@/lib/videoBoxCoords";

describe("videoBoxCoords", () => {
  it("round-trips source ↔ display for letterboxed contain", () => {
    const rect = getObjectFitContainRect(1080, 1920, 800, 600);
    expect(rect.scale).toBeCloseTo(Math.min(800 / 1080, 600 / 1920), 6);
    const src: [number, number, number, number] = [100, 200, 400, 500];
    const disp = sourceBoxToDisplayBox(src, rect);
    const back = displayBoxToSourceBox(disp, rect);
    expect(back[0]).toBeCloseTo(100, 4);
    expect(back[1]).toBeCloseTo(200, 4);
    expect(back[2]).toBeCloseTo(400, 4);
    expect(back[3]).toBeCloseTo(500, 4);
  });

  it("maps landscape contain with horizontal letterbox", () => {
    const rect = getObjectFitContainRect(1920, 1080, 400, 400);
    expect(rect.displayWidth).toBeCloseTo(400, 4);
    expect(rect.offsetY).toBeGreaterThan(0);
    expect(rect.offsetX).toBeCloseTo(0, 4);
  });

  it("converts client points into source space", () => {
    const rect = getObjectFitContainRect(100, 100, 200, 200);
    const container = { left: 10, top: 20, width: 200, height: 200 } as DOMRect;
    // Center of content at container (10+100, 20+100)
    const [sx, sy] = clientPointToSource(110, 120, container, rect);
    expect(sx).toBeCloseTo(50, 4);
    expect(sy).toBeCloseTo(50, 4);
  });

  it("clamps and hit-tests boxes", () => {
    expect(clampSourceBox([-10, -5, 50, 60], 40, 40)).toEqual([0, 0, 40, 40]);
    expect(pointInBox(5, 5, [0, 0, 10, 10])).toBe(true);
    expect(pointInBox(15, 5, [0, 0, 10, 10])).toBe(false);
  });

  it("title-cases labels and maps stages", () => {
    expect(titleCaseLabel("red backpack")).toBe("Red Backpack");
    expect(titleCaseLabel(null)).toBe("Object");
    expect(friendlyProcessingStage("tracking")).toBe("Following selected object");
    expect(friendlyProcessingStage("inpainting")).toBe("Rebuilding background");
    expect(friendlyProcessingStage("encoding_video")).toBe("Finalizing result");
    expect(friendlyProcessingStage("preparing_input")).toBe("Preparing video");
  });
});
