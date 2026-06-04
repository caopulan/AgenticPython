from __future__ import annotations

from dataclasses import dataclass, field

from .runtime import AgenticRunner, RunResult


@dataclass
class InteractiveSession:
    runner: AgenticRunner
    paused: bool = False
    finished: bool = False
    messages: list[str] = field(default_factory=list)

    def tick(self) -> bool:
        if self.finished:
            return False
        if self.paused:
            return True
        progressed = self.runner.step()
        if not progressed:
            self.finished = True
            self.runner.finish()
            self.messages.append(f"finished: artifacts at {self.runner.out_dir}")
            return False
        return True

    def submit_instruction(self, instruction: str) -> None:
        current = self.runner._current_instruction()
        if current is None:
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
        self.paused = True
        self.messages.append("instruction applied; paused until /resume")

    def pause(self) -> None:
        self.paused = True
        self.messages.append("paused")

    def resume(self) -> None:
        self.paused = False
        self.messages.append("resumed")

    def stop(self) -> RunResult:
        self.finished = True
        self.runner.stopped = True
        return self.runner.finish()
