/**
 * Real-phone difficult validation (IMG_5829 excerpt).
 * Similar-object: no dual-similar video in Send Anywhere — not covered here.
 */
import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const FIX = path.join("D:", "project", "artifacts", "real_video_tests");
const DIFF_32 = path.join(FIX, "phone_difficult_IMG_5829_32f.mp4");
const DIFF_16 = path.join(FIX, "phone_difficult_IMG_5829_16f.mp4");

async function confirmMask(page: Page) {
  await page.getByRole("button", { name: /Preview tracker mask/i }).click();
  await expect(
    page.getByRole("button", { name: /This is the object I want to track/i }),
  ).toBeVisible({ timeout: 420_000 });
  await page.getByRole("button", { name: /This is the object I want to track/i }).click();
}

async function selectTracker(page: Page, tracker: "edgetam" | "sam31") {
  const trackerName = tracker === "edgetam" ? "EdgeTAM" : "SAM 3.1";
  const tile = page.getByRole("radio", { name: trackerName, exact: true });
  await expect(tile).toBeVisible();
  await expect(tile.locator("span").filter({ hasText: /^Available$/i })).toBeVisible({
    timeout: 180_000,
  });
  await tile.click();
}

async function uploadAndFull(page: Page, file: string) {
  await page.goto("/studio");
  await page.locator('input[type="file"]').setInputFiles(file);
  await expect(page.locator("dd").first()).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText("540×960", { exact: true }).first()).toBeVisible({
    timeout: 60_000,
  });
  await page.getByRole("button", { name: /Full annotated video/i }).click();
}

async function pickCatOrBox(page: Page) {
  await page.getByPlaceholder(/person|backpack/i).fill("cat");
  await page.getByRole("button", { name: /Detect candidates/i }).click();
  try {
    await expect(page.getByLabel("Detection candidates").getByRole("button").first()).toBeVisible({
      timeout: 240_000,
    });
    await page.getByLabel("Detection candidates").getByRole("button").first().click();
  } catch {
    // Chest-held white cat region on 540x960 portrait
    for (const [i, v] of [
      ["x1", "160"],
      ["y1", "260"],
      ["x2", "420"],
      ["y2", "560"],
    ] as const) {
      await page.getByLabel(i).fill(v);
    }
  }
}

async function waitTerminal(page: Page) {
  await expect(page.locator('[aria-live="polite"]')).toContainText(
    /Job (succeeded|review_required|partial|failed)/i,
    { timeout: 30 * 60 * 1000 },
  );
}

test("REAL difficult phone IMG_5829 EdgeTAM browser", async ({ page }) => {
  test.skip(!fs.existsSync(DIFF_32), "difficult excerpt missing");
  await uploadAndFull(page, DIFF_32);
  await pickCatOrBox(page);
  await selectTracker(page, "edgetam");
  await confirmMask(page);
  const start = page.getByRole("button", { name: /Start Analysis/i }).first();
  await expect(start).toBeEnabled();
  await start.click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f-]{36}/, { timeout: 60_000 });
  const jobId = page.url().split("/").pop()!;
  await waitTerminal(page);
  await expect(page.getByText(/native_windows|Native/i).first()).toBeVisible();
  await expect(page.getByText(/^None$/).first()).toBeVisible({ timeout: 60_000 });
  console.log("REAL_DIFFICULT_EDGETAM_JOB", jobId);
});

test("REAL difficult phone IMG_5829 SAM31 browser", async ({ page }) => {
  test.skip(!fs.existsSync(DIFF_16), "difficult 16f excerpt missing");
  await page.goto("/system");
  await expect(page.getByText(/VisionForge-SAM31/).first()).toBeVisible({ timeout: 60_000 });
  await uploadAndFull(page, DIFF_16);
  await pickCatOrBox(page);
  await selectTracker(page, "sam31");
  await confirmMask(page);
  const start = page.getByRole("button", { name: /Start Analysis/i }).first();
  await expect(start).toBeEnabled();
  await start.click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f-]{36}/, { timeout: 60_000 });
  const jobId = page.url().split("/").pop()!;
  await waitTerminal(page);
  await expect(page.getByText(/wsl2|WSL/i).first()).toBeVisible();
  await expect(page.getByText(/^None$/).first()).toBeVisible({ timeout: 60_000 });
  console.log("REAL_DIFFICULT_SAM31_JOB", jobId);
});
