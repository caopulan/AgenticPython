from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

from .runtime import AgenticRunner, RunResult


@dataclass
class InteractiveSession:
    runner: AgenticRunner
    paused: bool = False
    finished: bool = False
    messages: list[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _busy: bool = field(default=False, init=False)
    _pending_instruction: str | None = field(default=None, init=False)
    _stop_requested: bool = field(default=False, init=False)
    _btw_threads: list[threading.Thread] = field(default_factory=list, init=False, repr=False)

    def tick(self) -> bool:
        with self._lock:
            if self.finished:
                return False
            if self._stop_requested:
                self._finish_locked(stopped=True)
                return False
            if self._busy:
                return True

            pending_instruction = self._pending_instruction
            if pending_instruction is not None:
                self._pending_instruction = None
                self._busy = True
                mode = "instruction"
            elif self.paused:
                return True
            else:
                self._busy = True
                mode = "step"

        progressed = True
        try:
            if mode == "instruction":
                self._apply_instruction(pending_instruction or "")
            else:
                progressed = self.runner.step()
        finally:
            with self._lock:
                self._busy = False
                if mode == "instruction":
                    self.paused = True
                    self.messages.append("instruction applied; paused until /resume")
                elif not progressed:
                    self._finish_locked(stopped=False)
                    return False
                if self._stop_requested:
                    self._finish_locked(stopped=True)
                    return False
        return True

    def submit_instruction(self, instruction: str) -> None:
        with self._lock:
            if self.finished:
                return
            self.paused = True
            if self._busy:
                self._pending_instruction = instruction
                self.messages.append("instruction queued; current instruction will finish first")
                return
            self._busy = True

        try:
            self._apply_instruction(instruction)
        finally:
            with self._lock:
                self._busy = False
                self.paused = True
                self.messages.append("instruction applied; paused until /resume")

    def queue_instruction(self, instruction: str) -> None:
        with self._lock:
            if self.finished:
                return
            self.paused = True
            self._pending_instruction = instruction
            if self._busy:
                self.messages.append("instruction queued; current instruction will finish first")
            else:
                self.messages.append("instruction queued for Codex")

    def submit_btw(self, question: str, visible_log: list[str] | None = None) -> None:
        with self._lock:
            if self.finished:
                self.messages.append("btw ignored; session is finished")
                return
            context = self._btw_context_locked(question, visible_log or [])
            thread = threading.Thread(
                target=self._run_btw,
                args=(context,),
                name="agenticpython-btw",
                daemon=True,
            )
            self._btw_threads.append(thread)
            self.messages.append("btw queued; execution continues")
            thread.start()

    def wait_for_btw(self, timeout: float | None = None) -> None:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            with self._lock:
                threads = list(self._btw_threads)
            if not threads:
                return
            for thread in threads:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                thread.join(timeout=remaining)
            with self._lock:
                self._btw_threads = [thread for thread in self._btw_threads if thread.is_alive()]
                if not self._btw_threads:
                    return
            if deadline is not None and time.monotonic() >= deadline:
                return

    def _btw_context_locked(self, question: str, visible_log: list[str]) -> dict[str, object]:
        return {
            "question": question,
            "paused": self.paused,
            "finished": self.finished,
            "runner_busy": self._busy,
            "visible_log": visible_log[-40:],
            "recent_events": list(self.runner.recent_events)[-20:],
        }

    def _run_btw(self, context: dict[str, object]) -> None:
        try:
            answer = self.runner.decision_client.ask_btw(context)
        except Exception as exc:
            with self._lock:
                self.messages.append(f"btw failed: {exc}")
            return
        with self._lock:
            self.messages.append(f"btw answer: {answer}")

    def _apply_instruction(self, instruction: str) -> None:
        self.runner.out_dir.mkdir(parents=True, exist_ok=True)
        current = self.runner._current_instruction()
        if current is None:
            with self._lock:
                self.finished = True
            self.runner.finish()
            return
        context = self.runner._context(
            trigger="human",
            current=current,
            user_instruction=instruction,
        )
        self.runner._record({"event": "trigger", "trigger": "human", "context": context})
        self.runner._apply_action(self.runner.decision_client.decide(context), current)

    def pause(self) -> None:
        with self._lock:
            self.paused = True
            if self._busy:
                self.messages.append("pause requested; current instruction will finish first")
            else:
                self.messages.append("paused")

    def resume(self) -> None:
        with self._lock:
            self.paused = False
            self.messages.append("resumed")

    def stop(self) -> RunResult:
        with self._lock:
            self._stop_requested = True
            self.paused = True
            if self._busy:
                self.runner.stopped = True
                self.messages.append("stop requested; current instruction will finish first")
                return RunResult(namespace=self.runner.namespace, out_dir=self.runner.out_dir, stopped=True)
            self._finish_locked(stopped=True)
            return RunResult(namespace=self.runner.namespace, out_dir=self.runner.out_dir, stopped=True)

    def drain_messages(self) -> list[str]:
        with self._lock:
            messages = list(self.messages)
            self.messages.clear()
            return messages

    def _finish_locked(self, stopped: bool) -> None:
        self.finished = True
        self.runner.stopped = stopped
        self.runner.finish()
        self.messages.append(f"finished: artifacts at {self.runner.out_dir}")


class BackgroundTicker:
    def __init__(self, session: InteractiveSession, delay: float = 0.05) -> None:
        self.session = session
        self.delay = delay
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="agenticpython-tui-ticker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 0.2) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self.session.tick():
                return
            time.sleep(self.delay)
