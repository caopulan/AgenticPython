# CPython Frame Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first robust native-backend foundation for AgenticPython: a versioned frame-event protocol, local controller boundary, reproducible CPython patch workflow, and a smoke path that proves package-internal Python frame observation.

**Architecture:** Keep CPython as a thin probe and keep Agent policy in AgenticPython. Runtime processes emit bounded JSONL frame events over a local transport; the controller validates commands and journals everything. The existing tape backend remains unchanged and continues to handle editable source-level scripts.

**Tech Stack:** Python 3.12-compatible stdlib, pytest, Unix domain sockets, newline-delimited JSON, CPython 3.12.x patchset managed outside the repo build cache.

---

## File Structure

- Create `src/agenticpython/native/__init__.py`: public native-backend exports.
- Create `src/agenticpython/native/protocol.py`: versioned event and command dataclasses, JSON encode/decode, value summarization.
- Create `tests/test_native_protocol.py`: deterministic protocol tests.
- Modify `src/agenticpython/cli.py`: add a protocol smoke command after protocol code is stable.
- Modify `.gitignore`: ignore `.agentpython-build/`.
- Create `tools/cpython_backend/fetch_cpython.py`: clone/fetch a pinned CPython tag into `.agentpython-build/cpython`.
- Create `tools/cpython_backend/apply_patches.py`: apply patch files from `runtimes/cpython/patches/`.
- Create `runtimes/cpython/README.md`: describe supported CPython tag, build commands, and smoke expectations.
- Create `runtimes/cpython/patches/0001-agentic-frame-probe.patch`: first runtime probe patch.
- Create `examples/native_package_demo/`: tiny imported package used to prove package-internal frame events.
- Create `tests/test_cpython_backend_tools.py`: tests for source-management commands that do not require network by default.

## Task 1: Native Protocol Foundation

**Files:**
- Create: `src/agenticpython/native/__init__.py`
- Create: `src/agenticpython/native/protocol.py`
- Test: `tests/test_native_protocol.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_native_protocol.py -q`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'agenticpython.native'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/agenticpython/native/__init__.py`:

```python
from .protocol import (
    PROTOCOL_VERSION,
    ControlCommand,
    FrameEvent,
    ProtocolError,
    decode_event,
    encode_command,
    summarize_value,
)

__all__ = [
    "PROTOCOL_VERSION",
    "ControlCommand",
    "FrameEvent",
    "ProtocolError",
    "decode_event",
    "encode_command",
    "summarize_value",
]
```

Create `src/agenticpython/native/protocol.py`:

```python
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
    if command.payload:
        data["payload"] = command.payload
    elif command.payload == {}:
        data["payload"] = {}
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_native_protocol.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Run full tests**

Run: `.venv/bin/python -m pytest -q`

Expected: all existing tests pass plus the new protocol tests.

- [ ] **Step 6: Commit**

```bash
git add src/agenticpython/native tests/test_native_protocol.py
git commit -m "feat: add native frame protocol foundation"
```

## Task 2: Protocol Smoke CLI

**Files:**
- Modify: `src/agenticpython/cli.py`
- Test: `tests/test_native_protocol.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_native_protocol.py`:

```python
from agenticpython.cli import _native_protocol_smoke


def test_native_protocol_smoke_returns_decoded_event():
    payload = _native_protocol_smoke()

    assert payload["event"] == "line"
    assert payload["command"] == "resume"
    assert payload["frame_id"] == "frame-smoke"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_native_protocol.py::test_native_protocol_smoke_returns_decoded_event -q`

Expected: FAIL with `ImportError` or `AttributeError` for `_native_protocol_smoke`.

- [ ] **Step 3: Implement the smoke helper and CLI command**

Add imports to `src/agenticpython/cli.py`:

```python
from .native.protocol import (
    PROTOCOL_VERSION,
    ControlCommand,
    decode_event,
    encode_command,
)
```

Add parser registration in `main()`:

```python
    subparsers.add_parser("native-protocol-smoke", help="Exercise the native frame protocol locally")
```

Add dispatch in `main()`:

```python
    elif args.command == "native-protocol-smoke":
        print(json.dumps(_native_protocol_smoke(), indent=2))
```

Add helper:

```python
def _native_protocol_smoke() -> dict[str, Any]:
    event = decode_event(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "event": "line",
                "run_id": "smoke",
                "process_id": 1,
                "frame_id": "frame-smoke",
                "filename": "smoke.py",
                "function": "<module>",
                "lineno": 1,
                "locals": {"x": "1"},
            }
        )
    )
    command = encode_command(
        ControlCommand(
            command="resume",
            run_id=event.run_id,
            frame_id=event.frame_id,
            reason="smoke complete",
        )
    )
    return {
        "event": event.event,
        "command": json.loads(command)["command"],
        "frame_id": event.frame_id,
    }
```

- [ ] **Step 4: Verify CLI**

Run: `.venv/bin/python -m pytest tests/test_native_protocol.py -q`

Expected: all protocol tests pass.

Run: `.venv/bin/agentpython native-protocol-smoke`

Expected JSON contains `"event": "line"`, `"command": "resume"`, and `"frame_id": "frame-smoke"`.

- [ ] **Step 5: Commit**

```bash
git add src/agenticpython/cli.py tests/test_native_protocol.py
git commit -m "feat: add native protocol smoke command"
```

## Task 3: CPython Source Management Skeleton

**Files:**
- Modify: `.gitignore`
- Create: `tools/cpython_backend/fetch_cpython.py`
- Create: `tools/cpython_backend/apply_patches.py`
- Create: `runtimes/cpython/README.md`
- Test: `tests/test_cpython_backend_tools.py`

- [ ] **Step 1: Write tests for local command construction**

Create `tests/test_cpython_backend_tools.py`:

```python
from pathlib import Path

from tools.cpython_backend.fetch_cpython import cpython_source_dir, build_fetch_commands
from tools.cpython_backend.apply_patches import patch_files


def test_cpython_source_dir_lives_in_ignored_build_cache(tmp_path):
    assert cpython_source_dir(tmp_path) == tmp_path / ".agentpython-build" / "cpython"


def test_fetch_commands_pin_cpython_312_tag(tmp_path):
    commands = build_fetch_commands(tmp_path, tag="v3.12.13")

    assert commands[0] == [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "v3.12.13",
        "https://github.com/python/cpython.git",
        str(tmp_path / ".agentpython-build" / "cpython"),
    ]


def test_patch_files_are_sorted(tmp_path):
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "0002-second.patch").write_text("second", encoding="utf-8")
    (patch_dir / "0001-first.patch").write_text("first", encoding="utf-8")

    assert patch_files(patch_dir) == [
        patch_dir / "0001-first.patch",
        patch_dir / "0002-second.patch",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cpython_backend_tools.py -q`

Expected: FAIL because `tools.cpython_backend` does not exist.

- [ ] **Step 3: Implement source-management helpers**

Create `tools/cpython_backend/fetch_cpython.py`:

```python
from __future__ import annotations

from pathlib import Path
import subprocess


def cpython_source_dir(repo_root: Path) -> Path:
    return repo_root / ".agentpython-build" / "cpython"


def build_fetch_commands(repo_root: Path, *, tag: str = "v3.12.13") -> list[list[str]]:
    destination = cpython_source_dir(repo_root)
    if destination.exists():
        return [["git", "-C", str(destination), "fetch", "--tags", "--depth", "1", "origin", tag]]
    return [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            "https://github.com/python/cpython.git",
            str(destination),
        ]
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cpython_source_dir(repo_root).parent.mkdir(parents=True, exist_ok=True)
    for command in build_fetch_commands(repo_root):
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
```

Create `tools/cpython_backend/apply_patches.py`:

```python
from __future__ import annotations

from pathlib import Path
import subprocess

from .fetch_cpython import cpython_source_dir


def patch_files(patch_dir: Path) -> list[Path]:
    return sorted(patch_dir.glob("*.patch"))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = cpython_source_dir(repo_root)
    for patch_file in patch_files(repo_root / "runtimes" / "cpython" / "patches"):
        subprocess.run(["git", "-C", str(source_dir), "apply", str(patch_file)], check=True)


if __name__ == "__main__":
    main()
```

Append to `.gitignore`:

```text
.agentpython-build/
```

Create `runtimes/cpython/README.md`:

```markdown
# Agentic CPython Runtime

AgenticPython keeps CPython source outside the repository. Fetch the pinned
source into `.agentpython-build/cpython`:

```bash
.venv/bin/python tools/cpython_backend/fetch_cpython.py
.venv/bin/python -m tools.cpython_backend.apply_patches
```

The first supported tag is `v3.12.13`.
```

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_cpython_backend_tools.py -q`

Expected: `3 passed`.

Run: `.venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add .gitignore runtimes/cpython/README.md tools/cpython_backend tests/test_cpython_backend_tools.py
git commit -m "feat: add cpython backend source tooling"
```

## Task 4: Controller Event Journal Skeleton

**Files:**
- Create: `src/agenticpython/native/controller.py`
- Test: `tests/test_native_controller.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_native_controller.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_native_controller.py -q`

Expected: FAIL because `agenticpython.native.controller` does not exist.

- [ ] **Step 3: Implement journal skeleton**

Create `src/agenticpython/native/controller.py`:

```python
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
    return {key: value for key, value in data.items() if value is not None and value != {}}
```

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_native_controller.py -q`

Expected: `1 passed`.

Run: `.venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agenticpython/native/controller.py tests/test_native_controller.py
git commit -m "feat: add native event journal"
```

## Task 5: First CPython Probe Patch

**Files:**
- Create: `runtimes/cpython/patches/0001-agentic-frame-probe.patch`
- Create: `examples/native_package_demo/pkgdemo/__init__.py`
- Create: `examples/native_package_demo/pkgdemo/inner.py`
- Create: `examples/native_package_demo/run_demo.py`

- [ ] **Step 1: Create the toy package demo**

Create `examples/native_package_demo/pkgdemo/__init__.py`:

```python
from .inner import compute

__all__ = ["compute"]
```

Create `examples/native_package_demo/pkgdemo/inner.py`:

```python
def compute(value: int) -> int:
    adjusted = value + 1
    return adjusted * 2
```

Create `examples/native_package_demo/run_demo.py`:

```python
from pkgdemo import compute


print(f"result={compute(3)}")
```

- [ ] **Step 2: Add a minimal patch**

Create `runtimes/cpython/patches/0001-agentic-frame-probe.patch` with the
smallest reviewable eval-loop change for CPython `v3.12.13`. The patch must:

- compile without `PYTHON_AGENTIC`;
- read `PYTHON_AGENTIC` and `PYTHON_AGENTIC_EVENTS` from the environment;
- emit JSONL frame events for call/line/return/exception to file descriptor 2
  or `PYTHON_AGENTIC_EVENTS`;
- avoid importing Python modules from the eval loop;
- include filename, function, line number, process id, and frame pointer-derived
  frame id.

- [ ] **Step 3: Build and smoke manually**

Run:

```bash
.venv/bin/python tools/cpython_backend/fetch_cpython.py
.venv/bin/python -m tools.cpython_backend.apply_patches
cd .agentpython-build/cpython
./configure --prefix="$PWD/../install-agentic"
make -j4
make install
PYTHON_AGENTIC=1 PYTHON_AGENTIC_EVENTS=/tmp/agentic-events.jsonl ../install-agentic/bin/python ../../examples/native_package_demo/run_demo.py
```

Expected:

- stdout contains `result=8`;
- `/tmp/agentic-events.jsonl` contains at least one event whose filename ends
  with `pkgdemo/inner.py` and function is `compute`.

- [ ] **Step 4: Commit**

```bash
git add runtimes/cpython/patches/0001-agentic-frame-probe.patch examples/native_package_demo
git commit -m "feat: add first cpython frame probe patch"
```

## Self-Review

- Spec coverage: protocol, controller boundary, build-cache policy, first CPython
  observation milestone, and package-internal demo all have tasks.
- Placeholder scan: the only deliberately non-inline code is the CPython C patch
  body, because it must be authored against the fetched `v3.12.13` source and
  reviewed as a real diff. Its required behavior, files, commands, and
  acceptance are explicit.
- Type consistency: protocol names are `FrameEvent`, `ControlCommand`,
  `decode_event`, `encode_command`, `summarize_value`, and
  `NativeEventJournal`; later tasks use those names exactly.
