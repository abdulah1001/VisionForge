import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIX = path.join("D:", "project", "artifacts", "real_video_tests");
const SIMPLE_MP4 = path.join(FIX, "simple.mp4");

test.describe.configure({ mode: "default" });

test("landing pauses WebGL when GPU busy and system page", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Track what matters/i })).toBeVisible();
  const src = fs.readFileSync(
    path.join(__dirname, "..", "src", "pages", "LandingPage.tsx"),
    "utf8",
  );
  expect(src).toMatch(/paused=\{Boolean\(gpuBusy\)\}/);
  await page.goto("/system");
  await expect(page.getByText(/Maximum active GPU jobs/i)).toBeVisible();
});

test("remover store starts with no selection", async () => {
  const store = fs.readFileSync(
    path.join(__dirname, "..", "src", "store", "removerStore.ts"),
    "utf8",
  );
  expect(store).toContain("box: null");
  expect(store.replace(/\s/g, "")).not.toContain("20,60,60,100");
});

test("studio upload shows video workspace and disables remove without selection", async ({
  page,
}) => {
  test.skip(!fs.existsSync(SIMPLE_MP4), "simple.mp4 missing");
  await page.goto("/studio");
  await expect(page.getByRole("heading", { name: /Object Remover Studio/i })).toBeAttached();
  await page.locator('input[type="file"]').setInputFiles(SIMPLE_MP4);
  await expect(page.locator("video").first()).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("remove-object-btn")).toBeDisabled();
  await expect(page.getByRole("button", { name: /Analyze Objects/i })).toBeVisible();
});

test("select manually enables remove object", async ({ page }) => {
  test.skip(!fs.existsSync(SIMPLE_MP4), "simple.mp4 missing");
  await page.goto("/studio");
  await page.locator('input[type="file"]').setInputFiles(SIMPLE_MP4);
  await expect(page.locator("video").first()).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: /Select manually/i }).click();
  const overlay = page.getByLabel("Detected objects");
  await expect(overlay).toBeVisible();
  const box = await overlay.boundingBox();
  expect(box).toBeTruthy();
  if (!box) return;
  await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.35);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.65);
  await page.mouse.up();
  await expect(page.getByText(/Selected for removal/i)).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId("remove-object-btn")).toBeEnabled();
});

test("replace video clears workspace selection", async ({ page }) => {
  test.skip(!fs.existsSync(SIMPLE_MP4), "simple.mp4 missing");
  await page.goto("/studio");
  await page.locator('input[type="file"]').setInputFiles(SIMPLE_MP4);
  await expect(page.locator("video").first()).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: /Select manually/i }).click();
  const overlay = page.getByLabel("Detected objects");
  const box = await overlay.boundingBox();
  if (box) {
    await page.mouse.move(box.x + 40, box.y + 40);
    await page.mouse.down();
    await page.mouse.move(box.x + 120, box.y + 120);
    await page.mouse.up();
  }
  await page.getByRole("button", { name: /Replace video/i }).click();
  await expect(page.getByText(/Upload a video/i)).toBeVisible();
  await page.locator('input[type="file"]').setInputFiles(SIMPLE_MP4);
  await expect(page.locator("video").first()).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("remove-object-btn")).toBeDisabled();
});

test("reduced motion preference does not crash studio", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/studio");
  await expect(page.getByRole("heading", { name: /Object Remover Studio/i })).toBeAttached();
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Track what matters/i })).toBeVisible();
});

const VIEWPORTS = [
  { w: 1920, h: 1080 },
  { w: 1440, h: 900 },
  { w: 1366, h: 768 },
  { w: 1024, h: 768 },
  { w: 768, h: 1024 },
  { w: 412, h: 892 },
  { w: 390, h: 844 },
  { w: 360, h: 800 },
] as const;

for (const vp of VIEWPORTS) {
  test(`responsive studio ${vp.w}x${vp.h} no horizontal overflow`, async ({ page }) => {
    await page.setViewportSize({ width: vp.w, height: vp.h });
    await page.goto("/studio");
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 2);
    await expect(page.getByRole("heading", { name: /Object Remover Studio/i })).toBeAttached();
  });
}

// Legacy track_analyze Studio GPU E2E suites lived here; Studio is now the object remover.
// Keep Jobs / System GPU coverage in other e2e specs as applicable.
