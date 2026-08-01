import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { api, ApiError } from "@/api/client";

const server = setupServer(
  http.post("/api/v1/jobs/j1/cancel", () =>
    HttpResponse.json({ job_id: "j1", status: "cancelling" }),
  ),
  http.get("/api/v1/jobs/missing", () =>
    HttpResponse.json(
      { error: { code: "JOB_NOT_FOUND", message: "Unknown job" } },
      { status: 404 },
    ),
  ),
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

it("requests cancellation", async () => {
  await expect(api.cancelJob("j1")).resolves.toMatchObject({
    job_id: "j1",
    status: "cancelling",
  });
});

it("parses structured API errors", async () => {
  await expect(api.job("missing")).rejects.toMatchObject({
    code: "JOB_NOT_FOUND",
    message: "Unknown job",
  });
  await expect(api.job("missing")).rejects.toBeInstanceOf(ApiError);
});
