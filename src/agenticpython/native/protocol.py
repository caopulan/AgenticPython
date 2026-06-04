from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

PROTOCOL_VERSION = 1

EVENTS = {"call", "line", "return", "exception"}
COMMANDS = {
    "pause",
    "resume",
    "stop",
    "snapshot",
    "mutate_local",
    "replace_future_function",
}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class FrameEvent:
    event: str
    run_id: str
    process_id: int
    frame_id: str
    filename: str
    function: str
    lineno: int
    rank: int | None = None
    world_size: int | None = None
    parent_frame_id: str | None = None
    locals: dict[str, str] = field(default_factory=dict)
    globals: dict[str, str] = field(default_factory=dict)
    exception: dict[str, str] | None = None


@dataclass(frozen=True)
class ControlCommand:
    command: str
    run_id: str
    frame_id: str | None = None
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def decode_event(line: str) -> FrameEvent:
    data = json.loads(line)
    _require_protocol(data)
    event = _expect_str(data, "event")
    if event not in EVENTS:
        raise ProtocolError(f"Unsupported event: {event}")
    return FrameEvent(
        event=event,
        run_id=_expect_str(data, "run_id"),
        process_id=_expect_int(data, "process_id"),
        rank=_optional_int(data, "rank"),
        world_size=_optional_int(data, "world_size"),
        frame_id=_expect_str(data, "frame_id"),
        parent_frame_id=_optional_str(data, "parent_frame_id"),
        filename=_expect_str(data, "filename"),
        function=_expect_str(data, "function"),
        lineno=_expect_int(data, "lineno"),
        locals=_string_map(data.get("locals", {}), "locals"),
        globals=_string_map(data.get("globals", {}), "globals"),
        exception=_optional_string_map(data, "exception"),
    )


def encode_command(command: ControlCommand) -> str:
    if command.command not in COMMANDS:
        raise ProtocolError(f"Unsupported command: {command.command}")
    data: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "command": command.command,
        "run_id": command.run_id,
    }
    if command.frame_id is not None:
        data["frame_id"] = command.frame_id
    if command.reason is not None:
        data["reason"] = command.reason
    data["payload"] = command.payload
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def summarize_value(value: Any, *, max_chars: int = 120) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:
        return f"<repr failed: {type(exc).__name__}>"
    if len(rendered) <= max_chars:
        return rendered
    if max_chars <= 3:
        return "." * max_chars
    return rendered[: max_chars - 3] + "..."


def _require_protocol(data: dict[str, Any]) -> None:
    version = data.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"Unsupported protocol_version: {version}")


def _expect_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ProtocolError(f"Expected string field: {key}")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProtocolError(f"Expected string field: {key}")
    return value


def _expect_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ProtocolError(f"Expected int field: {key}")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ProtocolError(f"Expected int field: {key}")
    return value


def _string_map(value: Any, key: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected object field: {key}")
    result: dict[str, str] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not isinstance(item_value, str):
            raise ProtocolError(f"Expected string map field: {key}")
        result[item_key] = item_value
    return result


def _optional_string_map(data: dict[str, Any], key: str) -> dict[str, str] | None:
    value = data.get(key)
    if value is None:
        return None
    return _string_map(value, key)
