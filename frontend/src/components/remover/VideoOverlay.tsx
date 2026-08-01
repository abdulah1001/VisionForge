import { useCallback, useEffect, useRef, useState } from "react";
import { Check } from "lucide-react";
import type { Box } from "@/api/types";
import type { DetectedObject } from "@/store/removerStore";
import {
  clampSourceBox,
  clientPointToSource,
  getObjectFitContainRect,
  sourceBoxToDisplayBox,
  titleCaseLabel,
  type BoxXyxy,
  type ContainRect,
} from "@/lib/videoBoxCoords";

export function VideoOverlay({
  naturalWidth,
  naturalHeight,
  candidates,
  selectedId,
  selectedBox,
  manualDraw,
  onSelect,
  onManualBox,
}: {
  naturalWidth: number;
  naturalHeight: number;
  candidates: DetectedObject[];
  selectedId: string | null;
  selectedBox: Box | null;
  manualDraw: boolean;
  onSelect: (id: string, box: Box, label: string | null) => void;
  onManualBox: (box: Box) => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [rect, setRect] = useState<ContainRect | null>(null);
  const [draft, setDraft] = useState<BoxXyxy | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);

  const measure = useCallback(() => {
    const el = rootRef.current;
    if (!el || naturalWidth < 1 || naturalHeight < 1) return;
    const { width, height } = el.getBoundingClientRect();
    setRect(getObjectFitContainRect(naturalWidth, naturalHeight, width, height));
  }, [naturalWidth, naturalHeight]);

  useEffect(() => {
    measure();
    const el = rootRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    return () => ro.disconnect();
  }, [measure]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (!manualDraw || !rect || !rootRef.current) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const bounds = rootRef.current.getBoundingClientRect();
    const [sx, sy] = clientPointToSource(e.clientX, e.clientY, bounds, rect);
    dragRef.current = { x: sx, y: sy };
    setDraft([sx, sy, sx, sy]);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!manualDraw || !dragRef.current || !rect || !rootRef.current) return;
    const bounds = rootRef.current.getBoundingClientRect();
    const [sx, sy] = clientPointToSource(e.clientX, e.clientY, bounds, rect);
    const { x, y } = dragRef.current;
    setDraft(
      clampSourceBox(
        [Math.min(x, sx), Math.min(y, sy), Math.max(x, sx), Math.max(y, sy)],
        naturalWidth,
        naturalHeight,
      ),
    );
  };

  const onPointerUp = () => {
    if (!manualDraw || !draft) {
      dragRef.current = null;
      return;
    }
    const [x1, y1, x2, y2] = draft;
    dragRef.current = null;
    setDraft(null);
    if (x2 - x1 >= 2 && y2 - y1 >= 2) {
      onManualBox(draft);
    }
  };

  const boxes: Array<{ key: string; box: BoxXyxy; label: string; selected: boolean }> = [];
  for (const c of candidates) {
    boxes.push({
      key: c.candidate_id,
      box: c.box_xyxy,
      label: titleCaseLabel(c.label),
      selected: c.candidate_id === selectedId,
    });
  }
  if (selectedBox && !selectedId) {
    boxes.push({
      key: "manual",
      box: selectedBox,
      label: titleCaseLabel("Selected region"),
      selected: true,
    });
  }
  if (draft) {
    boxes.push({
      key: "draft",
      box: draft,
      label: "Drawing…",
      selected: true,
    });
  }

  return (
    <div
      ref={rootRef}
      className={`absolute inset-0 z-10 touch-none ${
        manualDraw ? "pointer-events-auto cursor-crosshair" : "pointer-events-none"
      }`}
      role="listbox"
      aria-label="Detected objects"
      aria-multiselectable={false}
      onPointerDown={manualDraw ? onPointerDown : undefined}
      onPointerMove={manualDraw ? onPointerMove : undefined}
      onPointerUp={manualDraw ? onPointerUp : undefined}
      onPointerCancel={() => {
        dragRef.current = null;
        setDraft(null);
      }}
    >
      {rect &&
        boxes.map((b) => {
          const [dx1, dy1, dx2, dy2] = sourceBoxToDisplayBox(b.box, rect);
          const w = Math.max(1, dx2 - dx1);
          const h = Math.max(1, dy2 - dy1);
          return (
            <div
              key={b.key}
              role="option"
              aria-selected={b.selected}
              tabIndex={0}
              className={`pointer-events-auto absolute rounded-md border-2 transition ${
                b.selected
                  ? "border-[var(--success)] bg-[color-mix(in_srgb,var(--success)_14%,transparent)] shadow-[0_0_0_1px_color-mix(in_srgb,var(--success)_40%,transparent)]"
                  : "border-[var(--tracking)] bg-[color-mix(in_srgb,var(--tracking)_10%,transparent)]"
              }`}
              style={{ left: dx1, top: dy1, width: w, height: h }}
              onPointerDown={(e) => {
                if (manualDraw) return;
                e.stopPropagation();
                const c = candidates.find((x) => x.candidate_id === b.key);
                if (c) onSelect(c.candidate_id, c.box_xyxy, c.label);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  const c = candidates.find((x) => x.candidate_id === b.key);
                  if (c) onSelect(c.candidate_id, c.box_xyxy, c.label);
                }
              }}
            >
              <span
                className={`absolute -top-6 left-0 max-w-[min(100%,12rem)] truncate rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                  b.selected
                    ? "bg-[var(--success)] text-[#041018]"
                    : "bg-[var(--s1)] text-[var(--tracking)]"
                }`}
              >
                {b.label}
              </span>
              {b.selected && (
                <span className="absolute -right-2 -top-2 grid h-5 w-5 place-items-center rounded-full bg-[var(--success)] text-[#041018]">
                  <Check size={12} strokeWidth={3} aria-hidden />
                </span>
              )}
            </div>
          );
        })}
    </div>
  );
}
