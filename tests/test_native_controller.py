import json

from agenticpython.native.controller import NativeEventJournal
from agenticpython.native.protocol import PROTOCOL_VERSION


def test_native_event_journal_appends_decoded_events(tmp_path):
    journal = NativeEventJournal(tmp_path / "events.jsonl")
    journal.record_line(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "event": "call",
                "run_id": "run-1",
                "process_id": 7,
                "frame_id": "frame-1",
                "filename": "pkg.py",
                "function": "f",
                "lineno": 3,
            }
        )
    )

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]

    assert rows == [
        {
            "kind": "frame_event",
            "event": "call",
            "run_id": "run-1",
            "process_id": 7,
            "frame_id": "frame-1",
            "filename": "pkg.py",
            "function": "f",
            "lineno": 3,
        }
    ]
