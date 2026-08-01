import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import { isTerminal, type JobSpec } from "./types";

export const useCapabilities = () =>
  useQuery({
    queryKey: ["capabilities"],
    queryFn: api.capabilities,
    staleTime: 60_000,
  });

export const useReady = () =>
  useQuery({
    queryKey: ["ready"],
    queryFn: api.ready,
    refetchInterval: 30_000,
  });

export const useJobs = () =>
  useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: (q) =>
      q.state.data?.some((j) =>
        ["queued", "running", "cancelling"].includes(j.status),
      )
        ? 2500
        : false,
    refetchIntervalInBackground: false,
  });

export const useJob = (id: string) =>
  useQuery({
    queryKey: ["job", id],
    queryFn: () => api.job(id),
    enabled: Boolean(id),
    refetchInterval: (q) => {
      const job = q.state.data;
      if (!job || isTerminal(job.status) || document.hidden) return false;
      return 1500;
    },
    refetchIntervalInBackground: false,
  });

export const useResult = (id: string, enabled: boolean) =>
  useQuery({
    queryKey: ["result", id],
    queryFn: () => api.result(id),
    enabled,
  });

export const useArtifacts = (id: string, enabled: boolean) =>
  useQuery({
    queryKey: ["artifacts", id],
    queryFn: () => api.artifacts(id),
    enabled,
  });

export const useCreateJob = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, spec }: { file: File; spec: JobSpec }) =>
      api.createJob(file, spec),
    onSuccess: (job) => {
      localStorage.setItem("visionforge-active-job", job.job_id);
      void qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
};

export const useCancelJob = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.cancelJob(id),
    onSuccess: (body) => {
      void qc.invalidateQueries({ queryKey: ["job", body.job_id] });
      void qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
};
