from __future__ import annotations

from collections import deque
import curses
from pathlib import Path
import threading
import time
from typing import Any

from .codex_client import CodexSdkDecisionClient
from .interactive import BackgroundTicker, InteractiveSession
from .runtime import AgenticRunner
from .triggers import TriggerRule


_COLORS_READY = False


class TuiLog:
    def __init__(self, max_lines: int = 1000, log_level: str = "INFO") -> None:
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.log_level = log_level.upper()
        self._lock = threading.RLock()

    def append(self, line: str) -> None:
        with self._lock:
            for part in str(line).splitlines() or [""]:
                self.lines.append(part)

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.lines)

    def event_sink(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        if event_name == "execute":
            instruction = event.get("instruction", {})
            status = event.get("status")
            source = instruction.get("source", "")
            if self.log_level == "DEBUG":
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
    log_level: str = "INFO",
) -> None:
    log = TuiLog(log_level=log_level)
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
    _init_colors()
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    stdscr.nodelay(True)
    stdscr.keypad(True)
    input_text = ""
    log.append("Ready. Type /pause, /resume, /quit, or a natural-language instruction.")
    ticker = BackgroundTicker(session, delay=step_delay)
    ticker.start()

    try:
        while True:
            _drain_session_messages(session, log)
            _render(stdscr, session, log, input_text)

            try:
                key = stdscr.get_wch()
            except curses.error:
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
    finally:
        ticker.stop()


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
    log.append("queued instruction for Codex...")
    session.queue_instruction(text)
    _drain_session_messages(session, log)
    return False


def _drain_session_messages(session: InteractiveSession, log: TuiLog) -> None:
    for message in session.drain_messages():
        log.append(message)


def _render(stdscr: Any, session: InteractiveSession, log: TuiLog, input_text: str) -> None:
    height, width = stdscr.getmaxyx()
    stdscr.erase()
    _draw_header(stdscr, session, log, width)
    log_height = max(1, height - 5)
    visible = log.snapshot()[-log_height:]
    for row, line in enumerate(visible):
        stdscr.addnstr(row + 2, 1, line, max(0, width - 2))

    status = "paused" if session.paused else "running"
    if session.finished:
        status = "finished"
    divider = "─" * max(0, width - 1)
    stdscr.attron(_color_pair(3))
    stdscr.addnstr(height - 3, 0, divider, max(0, width - 1))
    stdscr.attroff(_color_pair(3))
    stdscr.addnstr(
        height - 2,
        0,
        f" {status.upper()}  out={session.runner.out_dir}  commands=/pause /resume /quit",
        max(0, width - 1),
        _color_pair(2),
    )
    prompt = "› "
    stdscr.addnstr(height - 1, 0, prompt + input_text, max(0, width - 1), _color_pair(4))
    stdscr.move(height - 1, min(width - 1, len(input_text) + 2))
    stdscr.refresh()


def _init_colors() -> None:
    global _COLORS_READY
    curses.start_color()
    if not curses.has_colors() or curses.COLOR_PAIRS <= 4:
        _COLORS_READY = False
        return
    background = -1
    try:
        curses.use_default_colors()
    except curses.error:
        background = curses.COLOR_BLACK
    curses.init_pair(1, curses.COLOR_CYAN, background)
    curses.init_pair(2, curses.COLOR_GREEN, background)
    curses.init_pair(3, curses.COLOR_BLUE, background)
    curses.init_pair(4, curses.COLOR_WHITE, background)
    _COLORS_READY = True


def _color_pair(index: int) -> int:
    if not _COLORS_READY:
        return 0
    return curses.color_pair(index)


def _draw_header(stdscr: Any, session: InteractiveSession, log: TuiLog, width: int) -> None:
    status = "paused" if session.paused else "running"
    if session.finished:
        status = "finished"
    title = f" AgenticPython  {status.upper()}  LOG={log.log_level} "
    stdscr.addnstr(0, 0, title.ljust(max(0, width - 1)), max(0, width - 1), _color_pair(1))
    subtitle = " Program log appears below. Natural language instructions pause and call Codex. "
    stdscr.addnstr(1, 0, subtitle.ljust(max(0, width - 1)), max(0, width - 1), _color_pair(3))
