"""Parse job-owned JSONL progress files safely."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def parse_progress_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("event") != "progress":
        return None
    if "stage" not in obj:
        return None
    return obj


def read_all_progress(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            ev = parse_progress_line(line)
            if ev is not None:
                events.append(ev)
    except OSError:
        return events
    return events


def tail_progress(
    path: Path,
    last_seen: dict[str, Any] | None,
) -> Iterator[dict[str, Any]]:
    """Yield new progress events after last_seen (by timestamp/stage tuple)."""
    events = read_all_progress(path)
    if not events:
        return
    if last_seen is None:
        yield events[-1]
        return
    # Yield events after matching last_seen identity
    seen = False
    for ev in events:
        if not seen:
            if (
                ev.get("timestamp") == last_seen.get("timestamp")
                and ev.get("stage") == last_seen.get("stage")
                and ev.get("completed") == last_seen.get("completed")
            ):
                seen = True
            continue
        yield ev
    if not seen and events:
        # last_seen not found — emit latest only
        yield events[-1]
