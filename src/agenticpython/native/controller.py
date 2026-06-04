from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .protocol import decode_event


class NativeEventJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_line(self, line: str) -> None:
        event = decode_event(line)
        row = {"kind": "frame_event", **_without_none(asdict(event))}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _without_none(data: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in data.items()
        if value is not None and value != {}
    }
