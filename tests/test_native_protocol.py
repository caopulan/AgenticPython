import json

import pytest

from agenticpython.native.protocol import (
    PROTOCOL_VERSION,
    ControlCommand,
    FrameEvent,
    ProtocolError,
    decode_event,
    encode_command,
    summarize_value,
)


def test_decodes_frame_event_with_rank_metadata():
    raw = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "event": "line",
            "run_id": "run-1",
            "process_id": 123,
            "rank": 0,
            "world_size": 2,
            "frame_id": "frame-abc",
            "parent_frame_id": "frame-parent",
            "filename": "/tmp/pkg/module.py",
            "function": "train_step",
            "lineno": 42,
            "locals": {"lr": "0.1", "step": "4"},
            "globals": {"__name__": "demo"},
        }
    )

    event = decode_event(raw)

    assert event == FrameEvent(
        event="line",
        run_id="run-1",
        process_id=123,
        rank=0,
        world_size=2,
        frame_id="frame-abc",
        parent_frame_id="frame-parent",
        filename="/tmp/pkg/module.py",
        function="train_step",
        lineno=42,
        locals={"lr": "0.1", "step": "4"},
        globals={"__name__": "demo"},
        exception=None,
    )


def test_rejects_unknown_protocol_version():
    raw = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION + 1,
            "event": "line",
            "run_id": "run-1",
            "process_id": 123,
            "frame_id": "frame-abc",
            "filename": "x.py",
            "function": "<module>",
            "lineno": 1,
        }
    )

    with pytest.raises(ProtocolError, match="Unsupported protocol_version"):
        decode_event(raw)


def test_encodes_control_command_without_null_fields():
    command = ControlCommand(
        command="resume",
        run_id="run-1",
        frame_id=None,
        reason="operator approved",
        payload={},
    )

    assert json.loads(encode_command(command)) == {
        "protocol_version": PROTOCOL_VERSION,
        "command": "resume",
        "run_id": "run-1",
        "reason": "operator approved",
        "payload": {},
    }


def test_summarize_value_is_bounded_and_stable_for_bad_repr():
    class BadRepr:
        def __repr__(self):
            raise RuntimeError("boom")

    long_value = "x" * 200

    assert summarize_value(long_value, max_chars=12) == "'xxxxxxxx..."
    assert summarize_value(BadRepr(), max_chars=80) == "<repr failed: RuntimeError>"
