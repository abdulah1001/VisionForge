import { expect, test } from "@playwright/test";

test.describe("visual screenshots", () => {
  test("landing carbon", async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem("visionforge-theme", "carbon"));
    await page.goto("/");
    await page.waitForTimeout(1500);
    await page.screenshot({
      path: "test-results/visual/landing-carbon.png",
      fullPage: true,
    });
  });

  test("landing porcelain", async ({ page }) => {
    await page.addInitScript(() =>
      localStorage.setItem("visionforge-theme", "porcelain"),
    );
    await page.goto("/");
    await page.waitForTimeout(800);
    await page.screenshot({
      path: "test-results/visual/landing-porcelain.png",
      fullPage: true,
    });
  });

  test("studio empty", async ({ page }) => {
    await page.goto("/studio");
    await page.screenshot({ path: "test-results/visual/studio-empty.png", fullPage: true });
  });

  test("jobs and system", async ({ page }) => {
    await page.goto("/jobs");
    await page.screenshot({ path: "test-results/visual/jobs.png", fullPage: true });
    await page.goto("/system");
    await expect(page.getByRole("heading", { name: "System" })).toBeVisible();
    await page.screenshot({ path: "test-results/visual/system.png", fullPage: true });
  });

  test("mobile landing", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    await page.screenshot({ path: "test-results/visual/mobile-landing.png", fullPage: true });
  });
});
