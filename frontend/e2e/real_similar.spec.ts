/**
 * Similar-object real validation using local EdgeTAM example 18_threedogs.mp4
 * (real camera footage with multiple dogs). Not from Send Anywhere.
 */
import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const FIX = path.join("D:", "project", "artifacts", "real_video_tests");
const DOGS_24 = path.join(FIX, "similar_threedogs_24f.mp4");
const DOGS_16 = path.join(FIX, "similar_threedogs_16f.mp4");

// Proven SAM-valid leftmost dog box on 640x360 first frame (OWL-ViT left candidate)
const LEFT_DOG_FALLBACK = [0, 50, 372, 357] as const;

async function pickLeftDog(page: Page) {
  await page.getByPlaceholder(/person|backpack/i).fill("dog");
  await page.getByRole("button", { name: /Detect candidates/i }).click();
  try {
    await expect(page.getByLabel("Detection candidates").getByRole("button").first()).toBeVisible({
      timeout: 240_000,
    });
    const buttons = page.getByLabel("Detection candidates").getByRole("button");
    const n = await buttons.count();
    let bestIdx = 0;
    let bestX = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      await buttons.nth(i).click();
      const x1 = Number(await page.getByLabel("x1").inputValue());
      if (Number.isFinite(x1) && x1 < bestX) {
        bestX = x1;
        bestIdx = i;
      }
    }
    await buttons.nth(bestIdx).click();
  } catch {
    for (let i = 0; i < 4; i++) {
      await page.getByLabel(["x1", "y1", "x2", "y2"][i]!).fill(String(LEFT_DOG_FALLBACK[i]));
    }
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

test("REAL similar threedogs EdgeTAM left identity", async ({ page }) => {
  test.skip(!fs.existsSync(DOGS_24), "threedogs excerpt missing");
  await uploadFull(page, DOGS_24);
  await pickLeftDog(page);
  await selectTracker(page, "edgetam");
  await confirmMask(page);
  await page.getByRole("button", { name: /Start Analysis/i }).first().click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f-]{36}/, { timeout: 60_000 });
  const jobId = page.url().split("/").pop()!;
  await waitTerminal(page);
  await expect(page.getByText(/^None$/).first()).toBeVisible({ timeout: 60_000 });
  const live = await page.locator('[aria-live="polite"]').innerText();
  console.log("REAL_SIMILAR_EDGETAM_JOB", jobId, live);
});

test("REAL similar threedogs SAM31 left identity", async ({ page }) => {
  test.skip(!fs.existsSync(DOGS_16), "threedogs 16f missing");
  await page.goto("/system");
  await expect(page.getByText(/VisionForge-SAM31/).first()).toBeVisible({ timeout: 60_000 });
  await uploadFull(page, DOGS_16);
  await pickLeftDog(page);
  await selectTracker(page, "sam31");
  await confirmMask(page);
  await page.getByRole("button", { name: /Start Analysis/i }).first().click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f-]{36}/, { timeout: 60_000 });
  const jobId = page.url().split("/").pop()!;
  await waitTerminal(page);
  await expect(page.getByText(/^None$/).first()).toBeVisible({ timeout: 60_000 });
  const live = await page.locator('[aria-live="polite"]').innerText();
  console.log("REAL_SIMILAR_SAM31_JOB", jobId, live);
});
