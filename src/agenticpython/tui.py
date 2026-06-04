from __future__ import annotations

from collections import deque
import curses
from pathlib import Path
import time
from typing import Any

from .codex_client import CodexSdkDecisionClient
from .interactive import InteractiveSession
from .runtime import AgenticRunner
from .triggers import TriggerRule


class TuiLog:
    def __init__(self, max_lines: int = 1000) -> None:
        self.lines: deque[str] = deque(maxlen=max_lines)

    def append(self, line: str) -> None:
        for part in str(line).splitlines() or [""]:
            self.lines.append(part)

    def event_sink(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        if event_name == "execute":
            instruction = event.get("instruction", {})
            status = event.get("status")
            source = instruction.get("source", "")
            self.append(f"[{status}] {instruction.get('id', '?')}: {source}")
            if event.get("stdout"):
                self.append(event["stdout"])
            if event.get("traceback"):
                self.append(event["traceback"].strip().splitlines()[-1])
            return
        if event_name == "trigger":
            self.append(f"[trigger:{event.get('trigger')}] paused for Codex")
            return
        if event_name == "patch":
            for operation in event.get("action", {}).get("operations", []):
                self.append(
                    "[patch] "
                    f"{operation.get('op')} {operation.get('target')}: "
                    f"{operation.get('code', '')}"
                )
            return
        self.append(str(event))


def run_tui(
    script_path: Path,
    triggers: list[TriggerRule],
    out_dir: Path,
    model: str | None = None,
    step_delay: float = 0.05,
) -> None:
    log = TuiLog()
    client = CodexSdkDecisionClient(model=model)
    try:
        runner = AgenticRunner.from_path(
            script_path,
            decision_client=client,
            triggers=triggers,
            out_dir=out_dir,
            echo_stdout=False,
            event_sink=log.event_sink,
        )
        session = InteractiveSession(runner)
        curses.wrapper(_curses_main, session, log, step_delay)
    finally:
        client.close()


def _curses_main(stdscr: Any, session: InteractiveSession, log: TuiLog, step_delay: float) -> None:
    curses.curs_set(1)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    input_text = ""
    log.append("AgenticPython TUI")
    log.append("Commands: /pause, /resume, /start, /quit, /help. Any other text is sent to Codex and pauses.")

    while True:
        _drain_session_messages(session, log)
        _render(stdscr, session, log, input_text)

        try:
            key = stdscr.get_wch()
        except curses.error:
            if not session.paused and not session.finished:
                session.tick()
            time.sleep(step_delay)
            continue

        if key in (curses.KEY_ENTER, "\n", "\r"):
            should_quit = _handle_input(input_text.strip(), session, log)
            input_text = ""
            if should_quit:
                return
            continue
        if key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
            input_text = input_text[:-1]
            continue
        if key in (27, "\x1b"):
            session.pause()
            continue
        if isinstance(key, str) and key.isprintable():
            input_text += key


def _handle_input(text: str, session: InteractiveSession, log: TuiLog) -> bool:
    if not text:
        return False

    normalized = text.strip().lower()
    if normalized in {"/quit", "/exit", ":q", "quit", "exit", "退出"}:
        result = session.stop()
        log.append(f"stopped: artifacts at {result.out_dir}")
        return True
    if normalized in {"/pause", "pause", "暂停"}:
        session.pause()
        _drain_session_messages(session, log)
        return False
    if normalized in {"/resume", "/start", "resume", "start", "继续", "开始"}:
        session.resume()
        _drain_session_messages(session, log)
        return False
    if normalized in {"/help", "help", "帮助"}:
        log.append("Use /pause to stop auto-run, /resume or /start to continue, /quit to exit.")
        log.append("Type any natural-language instruction to send current context to Codex; it stays paused afterward.")
        return False

    session.pause()
    log.append(f"[user] {text}")
    log.append("sending instruction to Codex...")
    session.submit_instruction(text)
    _drain_session_messages(session, log)
    return False


def _drain_session_messages(session: InteractiveSession, log: TuiLog) -> None:
    while session.messages:
        log.append(session.messages.pop(0))


def _render(stdscr: Any, session: InteractiveSession, log: TuiLog, input_text: str) -> None:
    height, width = stdscr.getmaxyx()
    stdscr.erase()
    log_height = max(1, height - 3)
    visible = list(log.lines)[-log_height:]
    for row, line in enumerate(visible):
        stdscr.addnstr(row, 0, line, max(0, width - 1))

    status = "paused" if session.paused else "running"
    if session.finished:
        status = "finished"
    divider = "-" * max(0, width - 1)
    stdscr.addnstr(height - 3, 0, divider, max(0, width - 1))
    stdscr.addnstr(
        height - 2,
        0,
        f"status={status} | out={session.runner.out_dir}",
        max(0, width - 1),
    )
    stdscr.addnstr(height - 1, 0, f"> {input_text}", max(0, width - 1))
    stdscr.move(height - 1, min(width - 1, len(input_text) + 2))
    stdscr.refresh()
