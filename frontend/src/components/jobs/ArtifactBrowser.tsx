import { useState } from "react";
import type { ArtifactItem } from "@/api/types";
import { api } from "@/api/client";

const GROUPS = [
  "manifest",
  "masks",
  "overlays",
  "crops",
  "similarity",
  "features",
  "timing",
  "logs",
] as const;

/** Frame dumps are huge — collapse by default so Jobs UI stays usable. */
const COLLAPSED_GROUPS = new Set(["masks", "overlays", "crops"]);
const PREVIEW_LIMIT = 12;

export function ArtifactBrowser({
  jobId,
  artifacts,
}: {
  jobId: string;
  artifacts: ArtifactItem[];
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const byGroup: { group: string; items: ArtifactItem[] }[] = GROUPS.map((g) => ({
    group: g,
    items: artifacts.filter((a) => a.group.toLowerCase().includes(g) || a.group === g),
  })).filter((g) => g.items.length > 0);

  const leftover = artifacts.filter(
    (a) =>
      !GROUPS.some(
        (g) => a.group.toLowerCase().includes(g) || a.group === g,
      ),
  );
  if (leftover.length) byGroup.push({ group: "other", items: leftover });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="display text-lg">Artifacts</h3>
        <span className="mono text-xs text-[var(--dim)]">{artifacts.length} items</span>
      </div>
      <p className="text-xs text-[var(--muted)]">
        Main deliverable is the annotated video + download ZIP. Masks / overlays /
        crops are per-frame dumps (one set per processed frame) — collapsed below.
      </p>
      {byGroup.map(({ group, items }) => {
        const collapse = COLLAPSED_GROUPS.has(group) && !expanded[group];
        const shown = collapse ? items.slice(0, PREVIEW_LIMIT) : items;
        return (
          <div key={group}>
            <div className="mb-2 flex items-center justify-between gap-2">
              <h4 className="eyebrow">
                {group}
                <span className="ml-2 normal-case tracking-normal text-[var(--dim)]">
                  {items.length}
                </span>
              </h4>
              {COLLAPSED_GROUPS.has(group) && items.length > PREVIEW_LIMIT && (
                <button
                  type="button"
                  className="text-xs text-[var(--tracking)] underline-offset-2 hover:underline"
                  onClick={() =>
                    setExpanded((e) => ({ ...e, [group]: !e[group] }))
                  }
                >
                  {expanded[group] ? "Show fewer" : `Show all ${items.length}`}
                </button>
              )}
            </div>
            <ul className="space-y-1">
              {shown.map((a) => (
                <li
                  key={a.id}
                  className="flex items-center gap-3 rounded-lg border border-[var(--b1)] bg-[var(--s2)] px-3 py-2 text-sm"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate">{a.name}</p>
                    <p className="mono text-[10px] text-[var(--dim)]">
                      {(a.size_bytes / 1024).toFixed(1)} KB
                    </p>
                  </div>
                  {a.preview_url ? (
                    <a
                      className="btn text-xs"
                      href={api.artifactUrl(jobId, a.id)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Preview
                    </a>
                  ) : null}
                </li>
              ))}
            </ul>
            {collapse && items.length > PREVIEW_LIMIT && (
              <p className="mt-2 text-[11px] text-[var(--dim)]">
                +{items.length - PREVIEW_LIMIT} more frame files in the ZIP download
              </p>
            )}
          </div>
        );
      })}
      {!artifacts.length && (
        <p className="text-sm text-[var(--muted)]">No allow-listed artifacts yet.</p>
      )}
    </div>
  );
}
