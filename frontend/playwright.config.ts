import { defineConfig, devices } from "@playwright/test";

const API = process.env.VF_API_URL ?? "http://127.0.0.1:8000";
const UI = process.env.VF_UI_URL ?? "http://127.0.0.1:5173";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  maxFailures: 0,
  timeout: 30 * 60 * 1000,
  expect: { timeout: 60_000 },
  use: {
    baseURL: UI,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  webServer: [
    {
      command:
        "D:\\project\\.venvs\\smoke\\Scripts\\python.exe -m visionforge.api.server --host 127.0.0.1 --port 8000",
      url: `${API}/health`,
      reuseExistingServer: true,
      timeout: 120_000,
      cwd: "D:\\project",
      env: {
        ...process.env,
        PYTHONPATH: "D:\\project",
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
      },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5173",
      url: UI,
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
