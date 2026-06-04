from __future__ import annotations

from collections import deque
import curses
from dataclasses import dataclass
from pathlib import Path
import threading
import textwrap
import time
from typing import Any

from .codex_client import CodexSdkDecisionClient
from .interactive import BackgroundTicker, InteractiveSession
from .runtime import AgenticRunner
from .triggers import TriggerRule


_READY_COLOR_PAIRS: set[int] = set()


@dataclass(frozen=True)
class LogEntry:
    text: str
    kind: str = "system"
    label: str | None = None


@dataclass(frozen=True)
class RenderedLogLine:
    text: str
    kind: str


_LABELS = {
    "agent": "AGENT",
    "error": "ERROR",
    "patch": "PATCH",
    "program": "OUT",
    "system": "STATE",
    "trace": "TRACE",
    "trigger": "CODEX",
    "user": "YOU",
}

_KIND_COLOR_PAIRS = {
    "agent": 5,
    "error": 8,
    "patch": 7,
    "program": 6,
    "system": 2,
    "trace": 3,
    "trigger": 7,
    "user": 5,
}


class TuiLog:
    def __init__(self, max_lines: int = 1000, log_level: str = "INFO") -> None:
        self.entries: deque[LogEntry] = deque(maxlen=max_lines)
        self.log_level = log_level.upper()
        self._lock = threading.RLock()

    def append(self, line: str, kind: str = "system", label: str | None = None) -> None:
        with self._lock:
            for part in str(line).splitlines() or [""]:
                self.entries.append(LogEntry(text=part, kind=kind, label=label))

    def snapshot(self) -> list[LogEntry]:
        with self._lock:
            return list(self.entries)

    def render_lines(self, width: int, max_lines: int) -> list[RenderedLogLine]:
        label_width = 9
        available = max(8, width - label_width - 1)
        rendered: list[RenderedLogLine] = []
        for entry in self.snapshot():
            label = (entry.label or _LABELS.get(entry.kind, entry.kind.upper()))[:label_width].ljust(label_width)
            prefix = f"{label} "
            continuation = " " * len(prefix)
            segments = _wrap_log_text(entry.text, available)
            for index, segment in enumerate(segments):
                line_prefix = prefix if index == 0 else continuation
                rendered.append(RenderedLogLine(text=f"{line_prefix}{segment}"[:width], kind=entry.kind))
        return rendered[-max_lines:]

    def event_sink(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        if event_name == "execute":
            instruction = event.get("instruction", {})
            status = event.get("status")
            source = instruction.get("source", "")
            if self.log_level == "DEBUG":
                self.append(f"{status} {instruction.get('id', '?')}: {source}", kind="trace")
            if event.get("stdout"):
                self.append(event["stdout"], kind="program")
            if event.get("traceback"):
                self.append(event["traceback"].strip().splitlines()[-1], kind="error")
            return
        if event_name == "trigger":
            self.append(f"{event.get('trigger')} context captured; waiting for Codex", kind="trigger")
            return
        if event_name == "patch":
            for operation in event.get("action", {}).get("operations", []):
                self.append(_operation_summary(operation), kind="patch")
            return
        self.append(str(event), kind="system")


def _wrap_log_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for physical_line in str(text).expandtabs(4).splitlines() or [""]:
        wrapped = textwrap.wrap(
            physical_line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            drop_whitespace=False,
            replace_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines


def _operation_summary(operation: dict[str, Any]) -> str:
    op = operation.get("op", "?")
    target = operation.get("target") or ""
    code = operation.get("code") or ""
    code_lines = [line.strip() for line in str(code).splitlines() if line.strip()]
    if not code_lines:
        return f"{op} {target}".strip()
    preview = code_lines[0]
    if len(preview) > 96:
        preview = f"{preview[:93]}..."
    line_count = f" ({len(code_lines)} lines)" if len(code_lines) > 1 else ""
    return f"{op} {target}{line_count}: {preview}".strip()


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
    log.append("Ready. Type /pause, /resume, /quit, or a natural-language instruction.", kind="system")
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
        log.append(f"stopped: artifacts at {result.out_dir}", kind="system")
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
        log.append("Use /pause to stop auto-run, /resume or /start to continue, /quit to exit.", kind="system")
        log.append(
            "Type any natural-language instruction to send current context to Codex; it stays paused afterward.",
            kind="system",
        )
        return False

    log.append(text, kind="user")
    session.queue_instruction(text)
    _drain_session_messages(session, log)
    return False


def _drain_session_messages(session: InteractiveSession, log: TuiLog) -> None:
    for message in session.drain_messages():
        log.append(message, kind=_session_message_kind(message))


def _session_message_kind(message: str) -> str:
    lowered = message.lower()
    if "instruction applied" in lowered:
        return "agent"
    if "queued" in lowered:
        return "trigger"
    return "system"


def _render(stdscr: Any, session: InteractiveSession, log: TuiLog, input_text: str) -> None:
    height, width = stdscr.getmaxyx()
    stdscr.erase()
    _draw_header(stdscr, session, log, width)
    log_height = max(1, height - 5)
    visible = log.render_lines(width=max(1, width - 2), max_lines=log_height)
    for row, line in enumerate(visible):
        stdscr.addnstr(row + 2, 1, line.text, max(0, width - 2), _entry_attr(line.kind))

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
    global _READY_COLOR_PAIRS
    _READY_COLOR_PAIRS = set()
    curses.start_color()
    color_pairs = getattr(curses, "COLOR_PAIRS", 0) or 0
    if not curses.has_colors() or color_pairs <= 1:
        return
    background = -1
    try:
        curses.use_default_colors()
    except curses.error:
        background = curses.COLOR_BLACK
    for index, color in (
        (1, curses.COLOR_CYAN),
        (2, curses.COLOR_GREEN),
        (3, curses.COLOR_BLUE),
        (4, curses.COLOR_WHITE),
        (5, curses.COLOR_MAGENTA),
        (6, curses.COLOR_CYAN),
        (7, curses.COLOR_YELLOW),
        (8, curses.COLOR_RED),
    ):
        if index >= color_pairs:
            continue
        try:
            curses.init_pair(index, color, background)
        except curses.error:
            continue
        _READY_COLOR_PAIRS.add(index)


def _color_pair(index: int) -> int:
    if index not in _READY_COLOR_PAIRS:
        return 0
    return curses.color_pair(index)


def _entry_attr(kind: str) -> int:
    return _color_pair(_KIND_COLOR_PAIRS.get(kind, 4))


def _draw_header(stdscr: Any, session: InteractiveSession, log: TuiLog, width: int) -> None:
    status = "paused" if session.paused else "running"
    if session.finished:
        status = "finished"
    title = f" AgenticPython  {status.upper()}  LOG={log.log_level} "
    stdscr.addnstr(0, 0, title.ljust(max(0, width - 1)), max(0, width - 1), _color_pair(1))
    subtitle = " Program log appears below. Natural language instructions pause and call Codex. "
    stdscr.addnstr(1, 0, subtitle.ljust(max(0, width - 1)), max(0, width - 1), _color_pair(3))
