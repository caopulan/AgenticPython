from __future__ import annotations

from collections import deque
import curses
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import site
import subprocess
import threading
import time
from typing import Any

from ..codex_client import CodexSdkDecisionClient
from ..tui import (
    TuiLog,
    _clip_to_cells,
    _color_pair,
    _entry_attr,
    _init_colors,
    _prompt_view,
)
from .protocol import FrameEvent, ProtocolError, decode_event


@dataclass(frozen=True)
class NativeRunPaths:
    out_dir: Path
    events_path: Path
    commands_path: Path
    pause_path: Path
    command_dir: Path


def make_native_run_paths(out_dir: Path) -> NativeRunPaths:
    return NativeRunPaths(
        out_dir=out_dir,
        events_path=out_dir / "frame_events.jsonl",
        commands_path=out_dir / "commands.jsonl",
        pause_path=out_dir / "PAUSED",
        command_dir=out_dir / "commands",
    )


def resolve_native_python(repo_root: Path, requested: Path | None = None) -> Path:
    if requested is not None:
        if not requested.exists():
            raise RuntimeError(f"Native Python does not exist: {requested}")
        return requested
    candidates = [
        repo_root / ".agentpython-build" / "cpython" / "python.exe",
        repo_root / ".agentpython-build" / "install-agentic" / "bin" / "python3.12",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Patched CPython runtime is missing. Build it with "
        "`.venv/bin/python -m tools.cpython_backend.fetch_cpython`, "
        "`.venv/bin/python -m tools.cpython_backend.apply_patches`, "
        "`cd .agentpython-build/cpython && ./configure --prefix=\"$PWD/../install-agentic\" && make -j4`."
    )


def build_native_env(
    base_env: dict[str, str],
    *,
    repo_root: Path,
    paths: NativeRunPaths,
    run_id: str,
    script_path: Path | None = None,
) -> dict[str, str]:
    env = dict(base_env)
    python_paths = [str(repo_root / "src")]
    python_paths.extend(site.getsitepackages())
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHON_AGENTIC"] = "1"
    env["PYTHON_AGENTIC_RUN_ID"] = run_id
    env["PYTHON_AGENTIC_EVENTS"] = str(paths.events_path)
    env["PYTHON_AGENTIC_COMMANDS"] = str(paths.commands_path)
    env["PYTHON_AGENTIC_PAUSE_FILE"] = str(paths.pause_path)
    if script_path is not None:
        env["PYTHON_AGENTIC_FILTER"] = str(script_path.resolve())
    return env


def write_exec_command(commands_path: Path, code_path: Path) -> None:
    with commands_path.open("a", encoding="utf-8") as handle:
        handle.write(f"exec_file\t{code_path}\n")


class NativeProcessSession:
    def __init__(
        self,
        *,
        script_path: Path,
        out_dir: Path,
        native_python: Path,
        repo_root: Path,
        log: TuiLog,
        model: str | None = None,
    ) -> None:
        self.script_path = script_path.resolve()
        self.out_dir = out_dir
        self.native_python = native_python
        self.repo_root = repo_root
        self.log = log
        self.model = model
        self.paths = make_native_run_paths(out_dir)
        self.run_id = f"native-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.paused = False
        self.finished = False
        self.returncode: int | None = None
        self.process: subprocess.Popen[str] | None = None
        self.recent_events: deque[FrameEvent] = deque(maxlen=80)
        self._threads: list[threading.Thread] = []
        self._client: CodexSdkDecisionClient | None = None
        self._client_lock = threading.Lock()
        self._command_index = 0
        self._state_lock = threading.RLock()

    def start(self) -> None:
        self.paths.command_dir.mkdir(parents=True, exist_ok=True)
        self.paths.commands_path.write_text("", encoding="utf-8")
        self.paths.events_path.write_text("", encoding="utf-8")
        if self.paths.pause_path.exists():
            self.paths.pause_path.unlink()
        env = build_native_env(
            os.environ,
            repo_root=self.repo_root,
            paths=self.paths,
            run_id=self.run_id,
            script_path=self.script_path,
        )
        self.process = subprocess.Popen(
            [str(self.native_python), str(self.script_path)],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.log.append(f"started native CPython pid={self.process.pid} python={self.native_python}", kind="system")
        self.log.append(f"events={self.paths.events_path}", kind="system")
        self._start_thread(self._read_stdout)
        self._start_thread(self._read_stderr)
        self._start_thread(self._tail_events)
        self._start_thread(self._wait_for_exit)

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._client is not None:
            self._client.close()

    def pause(self) -> None:
        self.paths.pause_path.write_text("paused\n", encoding="utf-8")
        with self._state_lock:
            self.paused = True
        self.log.append("pause requested; CPython will stop at the next trace safepoint", kind="system")

    def resume(self) -> None:
        if self.paths.pause_path.exists():
            self.paths.pause_path.unlink()
        with self._state_lock:
            self.paused = False
        self.log.append("resumed", kind="system")

    def stop(self) -> None:
        self.close()
        with self._state_lock:
            self.finished = True
        self.log.append(f"stopped: artifacts at {self.out_dir}", kind="system")

    def queue_code(self, code: str, *, source: str) -> Path:
        self.paths.command_dir.mkdir(parents=True, exist_ok=True)
        self._command_index += 1
        code_path = self.paths.command_dir / f"command-{self._command_index:04d}.py"
        code_path.write_text(code.rstrip() + "\n", encoding="utf-8")
        write_exec_command(self.paths.commands_path, code_path)
        self.log.append(f"queued native exec from {source}: {code_path.name}", kind="patch")
        preview = " ".join(line.strip() for line in code.splitlines() if line.strip())
        if preview:
            self.log.append(preview[:160], kind="patch")
        return code_path

    def submit_instruction(self, instruction: str, visible_log: list[str]) -> None:
        self.pause()
        self.log.append("asking Codex for native CPython code...", kind="trigger")
        thread = threading.Thread(
            target=self._decide_and_queue_code,
            args=(instruction, visible_log),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def submit_btw(self, question: str, visible_log: list[str]) -> None:
        thread = threading.Thread(
            target=self._answer_btw,
            args=(question, visible_log),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def _decide_and_queue_code(self, instruction: str, visible_log: list[str]) -> None:
        client = self._get_client()
        code = client.decide_native_code(
            {
                "user_instruction": instruction,
                "script": str(self.script_path),
                "recent_log": visible_log[-80:],
                "recent_frame_events": [_event_summary(event) for event in list(self.recent_events)[-40:]],
            }
        )
        self.queue_code(code, source="Codex")
        self.log.append("native instruction queued; still paused until /resume", kind="agent")

    def _answer_btw(self, question: str, visible_log: list[str]) -> None:
        client = self._get_client()
        answer = client.ask_btw(
            {
                "question": question,
                "script": str(self.script_path),
                "recent_log": visible_log[-80:],
                "recent_frame_events": [_event_summary(event) for event in list(self.recent_events)[-40:]],
            }
        )
        self.log.append(answer, kind="agent")

    def _get_client(self) -> CodexSdkDecisionClient:
        with self._client_lock:
            if self._client is None:
                self._client = CodexSdkDecisionClient(model=self.model)
            return self._client

    def _start_thread(self, target: Any) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.log.append(line.rstrip("\n"), kind="program")

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self.log.append(line.rstrip("\n"), kind="error")

    def _tail_events(self) -> None:
        offset = 0
        while not self.finished:
            if self.paths.events_path.exists():
                with self.paths.events_path.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for line in handle:
                        self._record_event_line(line)
                    offset = handle.tell()
            time.sleep(0.05)

    def _record_event_line(self, line: str) -> None:
        if not line.strip():
            return
        try:
            event = decode_event(line)
        except ProtocolError as exc:
            self.log.append(f"bad frame event: {exc}", kind="error")
            return
        self.recent_events.append(event)
        if self.log.log_level == "DEBUG" and event.filename.endswith(self.script_path.name):
            self.log.append(
                f"{event.event} {Path(event.filename).name}:{event.lineno} {event.function}",
                kind="trace",
            )

    def _wait_for_exit(self) -> None:
        assert self.process is not None
        self.returncode = self.process.wait()
        with self._state_lock:
            self.finished = True
            self.paused = False
        if self.paths.pause_path.exists():
            self.paths.pause_path.unlink()
        self.log.append(f"native process exited rc={self.returncode}", kind="system")


def run_native_tui(
    *,
    script_path: Path,
    out_dir: Path,
    native_python: Path | None = None,
    model: str | None = None,
    log_level: str = "INFO",
    step_delay: float = 0.05,
) -> None:
    repo_root = Path.cwd()
    resolved_python = resolve_native_python(repo_root, native_python)
    log = TuiLog(log_level=log_level)
    session = NativeProcessSession(
        script_path=script_path,
        out_dir=out_dir,
        native_python=resolved_python,
        repo_root=repo_root,
        log=log,
        model=model,
    )
    try:
        session.start()
        curses.wrapper(_native_curses_main, session, log, step_delay)
    finally:
        session.close()


def _native_curses_main(stdscr: Any, session: NativeProcessSession, log: TuiLog, step_delay: float) -> None:
    _init_colors()
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    stdscr.nodelay(True)
    stdscr.keypad(True)
    input_text = ""
    log.append("Native CPython TUI. Commands: /pause /resume /exec <code> /btw <question> /quit.", kind="system")

    while True:
        _render_native(stdscr, session, log, input_text)
        if session.finished:
            time.sleep(step_delay)
        try:
            key = stdscr.get_wch()
        except curses.error:
            time.sleep(step_delay)
            continue

        if key in (curses.KEY_ENTER, "\n", "\r"):
            should_quit = _handle_native_input(input_text.strip(), session, log)
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


def _handle_native_input(text: str, session: NativeProcessSession, log: TuiLog) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    if normalized in {"/quit", "/exit", ":q", "quit", "exit", "退出"}:
        session.stop()
        return True
    if normalized in {"/pause", "pause", "暂停"}:
        session.pause()
        return False
    if normalized in {"/resume", "/start", "resume", "start", "继续", "开始"}:
        session.resume()
        return False
    if normalized in {"/help", "help", "帮助"}:
        log.append("Use /exec <python code> for direct CPython-frame execution.", kind="system")
        log.append("Natural-language text pauses, asks Codex for Python code, queues it, and waits for /resume.", kind="system")
        return False
    if normalized == "/btw" or normalized.startswith("/btw "):
        question = text[4:].strip()
        if not question:
            log.append("Usage: /btw <side question>", kind="system")
            return False
        log.append(question, kind="btw")
        session.submit_btw(question, [line.text for line in log.render_lines(width=120, max_lines=80)])
        return False
    if normalized == "/exec" or normalized.startswith("/exec "):
        code = text[6:].strip()
        if not code:
            log.append("Usage: /exec <python code>", kind="system")
            return False
        session.pause()
        session.queue_code(code, source="operator /exec")
        log.append("direct code queued; still paused until /resume", kind="agent")
        return False

    log.append(text, kind="user")
    session.submit_instruction(text, [line.text for line in log.render_lines(width=120, max_lines=80)])
    return False


def _render_native(stdscr: Any, session: NativeProcessSession, log: TuiLog, input_text: str) -> None:
    height, width = stdscr.getmaxyx()
    stdscr.erase()
    _draw_native_header(stdscr, session, log, width)
    log_height = max(1, height - 5)
    visible = log.render_lines(width=max(1, width - 2), max_lines=log_height)
    for row, line in enumerate(visible):
        stdscr.addnstr(row + 2, 1, line.text, max(0, width - 2), _entry_attr(line.kind))

    status = "paused" if session.paused else "running"
    if session.finished:
        status = "finished"
    divider = "─" * max(0, width - 1)
    stdscr.attron(_color_pair(9))
    stdscr.addnstr(height - 3, 0, divider, max(0, width - 1))
    stdscr.attroff(_color_pair(9))
    stdscr.addnstr(
        height - 2,
        0,
        f" {status.upper()}  out={session.out_dir}  commands=/pause /resume /exec /btw /quit",
        max(0, width - 1),
        _color_pair(2),
    )
    prompt_line, cursor_col = _prompt_view(input_text, width)
    stdscr.addnstr(height - 1, 0, prompt_line, max(0, width - 1), _color_pair(4))
    stdscr.move(height - 1, min(width - 1, cursor_col))
    stdscr.refresh()


def _draw_native_header(stdscr: Any, session: NativeProcessSession, log: TuiLog, width: int) -> None:
    status = "paused" if session.paused else "running"
    if session.finished:
        status = "finished"
    title = f" AgenticPython Native CPython  {status.upper()}  LOG={log.log_level} "
    stdscr.addnstr(0, 0, title.ljust(max(0, width - 1)), max(0, width - 1), _color_pair(1))
    subtitle = " MNIST runs inside patched CPython. /exec queues code at the next frame safepoint. "
    stdscr.addnstr(1, 0, _clip_to_cells(subtitle.ljust(max(0, width - 1)), width - 1), max(0, width - 1), _color_pair(9))


def _event_summary(event: FrameEvent) -> dict[str, Any]:
    return {
        "event": event.event,
        "filename": event.filename,
        "function": event.function,
        "lineno": event.lineno,
    }
