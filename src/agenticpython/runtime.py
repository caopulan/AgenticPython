from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import contextlib
import io
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from .actions import AgentAction, DecisionClient, PatchOperation
from .parser import parse_script
from .tape import ForBlock, Instruction, ModuleTape, TapeItem
from .triggers import TriggerRule


@dataclass
class RunResult:
    namespace: dict[str, Any]
    out_dir: Path
    stopped: bool = False


@dataclass
class _MainFrame:
    items: list[TapeItem]
    index: int = 0


@dataclass
class _LoopFrame:
    block: ForBlock
    iterator: Any
    body_index: int = 0
    active: bool = False


class _CaptureProxy:
    def __init__(self, fallback: Any) -> None:
        self.fallback = fallback
        self.buffer: io.StringIO | None = None

    def write(self, text: str) -> int:
        if self.buffer is not None:
            return self.buffer.write(text)
        return self.fallback.write(text)

    def flush(self) -> None:
        if self.buffer is not None:
            return
        self.fallback.flush()

    def isatty(self) -> bool:
        return False


class AgenticRunner:
    def __init__(
        self,
        tape: ModuleTape,
        decision_client: DecisionClient,
        triggers: list[TriggerRule] | None = None,
        out_dir: str | Path | None = None,
        echo_stdout: bool = True,
        event_sink: Any | None = None,
    ) -> None:
        self.tape = tape
        self.decision_client = decision_client
        self.triggers = triggers or []
        self.out_dir = Path(out_dir or ".agentpython-runs/latest")
        self.namespace: dict[str, Any] = {"__name__": "__agentpython__"}
        self.frames: list[_MainFrame | _LoopFrame] = [_MainFrame(self.tape.items)]
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=50)
        self.stopped = False
        self._current_error: str | None = None
        self.echo_stdout = echo_stdout
        self.event_sink = event_sink
        self._stdout_proxy = _CaptureProxy(sys.stdout)
        self._stderr_proxy = _CaptureProxy(sys.stderr)

    @classmethod
    def from_path(
        cls,
        script_path: str | Path,
        decision_client: DecisionClient,
        triggers: list[TriggerRule] | None = None,
        out_dir: str | Path | None = None,
        echo_stdout: bool = True,
        event_sink: Any | None = None,
    ) -> "AgenticRunner":
        path = Path(script_path)
        return cls(
            parse_script(path.read_text(encoding="utf-8"), filename=str(path)),
            decision_client=decision_client,
            triggers=triggers,
            out_dir=out_dir,
            echo_stdout=echo_stdout,
            event_sink=event_sink,
        )

    def run(self) -> RunResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        while not self.stopped:
            if not self.step():
                break
        self._write_artifacts()
        return RunResult(namespace=self.namespace, out_dir=self.out_dir, stopped=self.stopped)

    def step(self) -> bool:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        current = self._current_instruction()
        if current is None:
            return False

        if self._current_error is None and self._handle_human_trigger(current):
            return True

        try:
            stdout = self._execute_instruction(current)
        except Exception:
            self._current_error = traceback.format_exc()
            self._record(
                {
                    "event": "execute",
                    "instruction": current.to_dict(),
                    "status": "error",
                    "traceback": self._current_error,
                }
            )
            self._handle_error_trigger(current)
            return True

        self._current_error = None
        self._record(
            {
                "event": "execute",
                "instruction": current.to_dict(),
                "status": "ok",
                "stdout": stdout,
            }
        )
        self._advance_current()
        return True

    def finish(self) -> RunResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_artifacts()
        return RunResult(namespace=self.namespace, out_dir=self.out_dir, stopped=self.stopped)

    def _current_instruction(self) -> Instruction | None:
        while self.frames:
            frame = self.frames[-1]
            if isinstance(frame, _MainFrame):
                if frame.index >= len(frame.items):
                    self.frames.pop()
                    continue
                item = frame.items[frame.index]
                if isinstance(item, Instruction):
                    return item
                iterator = iter(eval(item.iter_source, self.namespace, self.namespace))
                self.frames.append(_LoopFrame(block=item, iterator=iterator))
                continue

            if not frame.active or frame.body_index >= len(frame.block.body):
                try:
                    loop_value = next(frame.iterator)
                except StopIteration:
                    self.frames.pop()
                    if self.frames and isinstance(self.frames[-1], _MainFrame):
                        self.frames[-1].index += 1
                    continue
                self.namespace["__agentpython_loop_value__"] = loop_value
                exec(
                    compile(
                        f"{frame.block.target} = __agentpython_loop_value__",
                        f"<agentpython:{frame.block.id}:target>",
                        "exec",
                    ),
                    self.namespace,
                    self.namespace,
                )
                frame.active = True
                frame.body_index = 0
                if not frame.block.body:
                    frame.active = False
                    continue
            return frame.block.body[frame.body_index]
        return None

    def _advance_current(self) -> None:
        if not self.frames:
            return
        frame = self.frames[-1]
        if isinstance(frame, _MainFrame):
            frame.index += 1
            return
        frame.body_index += 1

    def _execute_instruction(self, instruction: Instruction) -> str:
        return self._execute_source(instruction.source, f"<agentpython:{instruction.id}>")

    def _execute_source(self, source: str, filename: str) -> str:
        stream = io.StringIO()
        self._stdout_proxy.buffer = stream
        self._stderr_proxy.buffer = stream
        try:
            with contextlib.redirect_stdout(self._stdout_proxy), contextlib.redirect_stderr(self._stderr_proxy):
                exec(
                    compile(source, filename, "exec"),
                    self.namespace,
                    self.namespace,
                )
        finally:
            self._stdout_proxy.buffer = None
            self._stderr_proxy.buffer = None
        stdout = stream.getvalue()
        if stdout and self.echo_stdout:
            sys.stdout.write(stdout)
        return stdout

    def _handle_human_trigger(self, current: Instruction) -> bool:
        for trigger in self.triggers:
            if not trigger.evaluate(self.namespace):
                continue
            context = self._context(
                trigger="human",
                current=current,
                user_instruction=trigger.instruction,
            )
            self._record({"event": "trigger", "trigger": "human", "context": context})
            self._apply_action(self.decision_client.decide(context), current)
            return True
        return False

    def _handle_error_trigger(self, current: Instruction) -> None:
        context = self._context(
            trigger="error",
            current=current,
            user_instruction=(
                "The current instruction failed. Replace only the failing instruction "
                "with a safe corrected instruction, then resume. If the failing source "
                "is `a = 1 / 0`, replace it with `a = 1 / 1`."
            ),
            error=self._current_error,
        )
        self._record({"event": "trigger", "trigger": "error", "context": context})
        self._apply_action(self.decision_client.decide(context), current)

    def _apply_action(self, action: AgentAction, current: Instruction) -> None:
        for operation in action.operations:
            self._apply_operation(operation, current)
        self._record({"event": "patch", "action": action.to_dict()})
        if not action.resume:
            self.stopped = True

    def _apply_operation(self, operation: PatchOperation, current: Instruction) -> None:
        op = operation.op
        target = current.id if operation.target == "__current__" else operation.target

        if op == "resume":
            return
        if op == "stop":
            self.stopped = True
            return
        if op == "execute_now":
            assert operation.code is not None
            stdout = self._execute_source(operation.code, "<agentpython:execute_now>")
            self._record(
                {
                    "event": "execute_now",
                    "operation": {
                        "op": operation.op,
                        "target": operation.target,
                        "code": operation.code,
                    },
                    "stdout": stdout,
                }
            )
            return

        assert target is not None
        container, index, target_instruction = self.tape.find_instruction(target)
        if op == "replace":
            assert operation.code is not None
            target_instruction.source = operation.code.strip("\n")
            return
        if op == "delete":
            del container[index]
            self._adjust_current_after_delete(container, index)
            return

        assert operation.code is not None
        new_instruction = self.tape.make_patch_instruction(operation.code, target_instruction)
        if op == "insert_before":
            container.insert(index, new_instruction)
            self._adjust_current_after_insert(container, index)
            return
        if op == "insert_after":
            container.insert(index + 1, new_instruction)
            return
        raise AssertionError(f"Unhandled operation: {op}")

    def _adjust_current_after_insert(self, container: list[Any], insert_index: int) -> None:
        frame = self.frames[-1] if self.frames else None
        if isinstance(frame, _MainFrame) and frame.items is container and insert_index <= frame.index:
            frame.index = insert_index
        if isinstance(frame, _LoopFrame) and frame.block.body is container and insert_index <= frame.body_index:
            frame.body_index = insert_index

    def _adjust_current_after_delete(self, container: list[Any], delete_index: int) -> None:
        frame = self.frames[-1] if self.frames else None
        if isinstance(frame, _MainFrame) and frame.items is container and delete_index < frame.index:
            frame.index -= 1
        if isinstance(frame, _LoopFrame) and frame.block.body is container and delete_index < frame.body_index:
            frame.body_index -= 1

    def _context(
        self,
        trigger: str,
        current: Instruction,
        user_instruction: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "trigger": trigger,
            "user_instruction": user_instruction,
            "current_instruction": current.to_dict(),
            "error": error,
            "namespace": _namespace_summary(self.namespace),
            "recent_events": list(self.recent_events),
            "future_tape": self._future_tape_window(current),
        }

    def _future_tape_window(self, current: Instruction, limit: int = 8) -> list[dict[str, Any]]:
        flattened: list[Instruction] = []
        for item in self.tape.items:
            if isinstance(item, Instruction):
                flattened.append(item)
            else:
                flattened.extend(item.body)
        try:
            start = next(index for index, instruction in enumerate(flattened) if instruction.id == current.id)
        except StopIteration:
            start = 0
        return [instruction.to_dict() for instruction in flattened[start : start + limit]]

    def _record(self, event: dict[str, Any]) -> None:
        self.recent_events.append(event)
        if self.event_sink is not None:
            self.event_sink(event)
        with (self.out_dir / "journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, default=repr) + "\n")

    def _write_artifacts(self) -> None:
        (self.out_dir / "final_tape.json").write_text(
            json.dumps(self.tape.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        (self.out_dir / "replay.py").write_text(self.tape.to_source(), encoding="utf-8")


def _namespace_summary(namespace: dict[str, Any]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for key, value in namespace.items():
        if key.startswith("__"):
            continue
        representation = repr(value)
        if len(representation) > 200:
            representation = representation[:197] + "..."
        summary[key] = representation
    return summary
