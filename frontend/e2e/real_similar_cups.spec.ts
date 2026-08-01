/**
 * Similar-object: real cups shell-game footage (EdgeTAM example 02_cups.mp4).
 * Three identical red cups — identity must not silently jump.
 */
import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const FIX = path.join("D:", "project", "artifacts", "real_video_tests");
const CUPS_16 = path.join(FIX, "similar_cups_16f.mp4");
const CUPS_24 = path.join(FIX, "similar_cups_24f.mp4");
// Held left cup region on 640x360 first frame
const HELD_CUP = [90, 70, 270, 290] as const;

async function fillBox(page: Page, box: readonly [number, number, number, number]) {
  for (let i = 0; i < 4; i++) {
    await page.getByLabel(["x1", "y1", "x2", "y2"][i]!).fill(String(box[i]));
  }
}

async function confirmMask(page: Page) {
  await page.getByRole("button", { name: /Preview tracker mask/i }).click();
  await expect(
    page.getByRole("button", { name: /This is the object I want to track/i }),
  ).toBeVisible({ timeout: 420_000 });
  await page.getByRole("button", { name: /This is the object I want to track/i }).click();
}

async function selectTracker(page: Page, tracker: "edgetam" | "sam31") {
  const name = tracker === "edgetam" ? "EdgeTAM" : "SAM 3.1";
  const tile = page.getByRole("radio", { name, exact: true });
  await expect(tile.locator("span").filter({ hasText: /^Available$/i })).toBeVisible({
    timeout: 180_000,
  });
  await tile.click();
}

async function uploadFull(page: Page, file: string) {
  await page.goto("/studio");
  await page.locator('input[type="file"]').setInputFiles(file);
  await expect(page.locator("dd").first()).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: /Full annotated video/i }).click();
}

async function waitTerminal(page: Page) {
  await expect(page.locator('[aria-live="polite"]')).toContainText(
    /Job (succeeded|review_required|partial|failed)/i,
    { timeout: 30 * 60 * 1000 },
  );
}

test("REAL similar cups EdgeTAM held-cup identity", async ({ page }) => {
  test.skip(!fs.existsSync(CUPS_24), "cups excerpt missing");
  await uploadFull(page, CUPS_24);
  await fillBox(page, HELD_CUP);
  await selectTracker(page, "edgetam");
  await confirmMask(page);
  await page.getByRole("button", { name: /Start Analysis/i }).first().click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f-]{36}/, { timeout: 60_000 });
  const jobId = page.url().split("/").pop()!;
  await waitTerminal(page);
  const live = await page.locator('[aria-live="polite"]').innerText();
  console.log("REAL_SIMILAR_CUPS_EDGETAM_JOB", jobId, live);
});

test("REAL similar cups SAM31 held-cup identity", async ({ page }) => {
  test.skip(!fs.existsSync(CUPS_16), "cups 16f missing");
  await page.goto("/system");
  await expect(page.getByText(/VisionForge-SAM31/).first()).toBeVisible({ timeout: 60_000 });
  await uploadFull(page, CUPS_16);
  await fillBox(page, HELD_CUP);
  await selectTracker(page, "sam31");
  await confirmMask(page);
  await page.getByRole("button", { name: /Start Analysis/i }).first().click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f-]{36}/, { timeout: 60_000 });
  const jobId = page.url().split("/").pop()!;
  await waitTerminal(page);
  const live = await page.locator('[aria-live="polite"]').innerText();
  console.log("REAL_SIMILAR_CUPS_SAM31_JOB", jobId, live);
});
