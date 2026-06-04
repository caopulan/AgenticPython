from __future__ import annotations

import ast
from collections import deque
import curses
from dataclasses import dataclass
from datetime import datetime
import json
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
    journal_path: Path
    pause_path: Path
    command_dir: Path


def make_native_run_paths(out_dir: Path) -> NativeRunPaths:
    return NativeRunPaths(
        out_dir=out_dir,
        events_path=out_dir / "frame_events.jsonl",
        commands_path=out_dir / "commands.jsonl",
        journal_path=out_dir / "journal.jsonl",
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
    env["PYTHONUNBUFFERED"] = "1"
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
        self._journal_lock = threading.RLock()

    def start(self) -> None:
        self.paths.command_dir.mkdir(parents=True, exist_ok=True)
        self.paths.commands_path.write_text("", encoding="utf-8")
        self.paths.events_path.write_text("", encoding="utf-8")
        self.paths.journal_path.write_text("", encoding="utf-8")
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
        self._write_journal(
            "session_start",
            {
                "run_id": self.run_id,
                "script": str(self.script_path),
                "native_python": str(self.native_python),
                "pid": self.process.pid,
                "events_path": str(self.paths.events_path),
                "commands_path": str(self.paths.commands_path),
                "journal_path": str(self.paths.journal_path),
                "python_agentic_filter": env.get("PYTHON_AGENTIC_FILTER"),
            },
        )
        self.log_line(f"started native CPython pid={self.process.pid} python={self.native_python}", kind="system")
        self.log_line(f"events={self.paths.events_path}", kind="system")
        self.log_line(f"journal={self.paths.journal_path}", kind="system")
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
        self.log_line(
            "pause requested; CPython will stop at the next trace safepoint",
            kind="system",
            event="pause",
        )

    def resume(self) -> None:
        if self.paths.pause_path.exists():
            self.paths.pause_path.unlink()
        with self._state_lock:
            self.paused = False
        self.log_line("resumed", kind="system", event="resume")

    def stop(self) -> None:
        self.close()
        with self._state_lock:
            self.finished = True
        self.log_line(f"stopped: artifacts at {self.out_dir}", kind="system", event="stop")

    def queue_code(self, code: str, *, source: str) -> Path:
        self.paths.command_dir.mkdir(parents=True, exist_ok=True)
        self._command_index += 1
        code_path = self.paths.command_dir / f"command-{self._command_index:04d}.py"
        code_path.write_text(code.rstrip() + "\n", encoding="utf-8")
        write_exec_command(self.paths.commands_path, code_path)
        self._write_journal(
            "command_queued",
            {
                "source": source,
                "code_path": str(code_path),
                "code": code,
            },
        )
        self.log_line(f"queued native exec from {source}: {code_path.name}", kind="patch")
        preview = " ".join(line.strip() for line in code.splitlines() if line.strip())
        if preview:
            self.log_line(preview[:160], kind="patch")
        return code_path

    def submit_instruction(self, instruction: str, visible_log: list[str]) -> None:
        self._write_journal("user_instruction", {"text": instruction})
        self.pause()
        self.log_line("asking Codex for native CPython code...", kind="trigger", event="codex_request_started")
        thread = threading.Thread(
            target=self._decide_and_queue_code,
            args=(instruction, visible_log),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def submit_btw(self, question: str, visible_log: list[str]) -> None:
        self._write_journal("btw_question", {"text": question})
        thread = threading.Thread(
            target=self._answer_btw,
            args=(question, visible_log),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def _decide_and_queue_code(self, instruction: str, visible_log: list[str]) -> None:
        client = self._get_client()
        context = self._build_decision_context(instruction, visible_log)
        self._write_journal("codex_request_context", context)
        code = client.decide_native_code(context)
        self._write_journal("codex_response_code", {"code": code})
        self.queue_code(code, source="Codex")
        self.log_line("native instruction queued; still paused until /resume", kind="agent")

    def _answer_btw(self, question: str, visible_log: list[str]) -> None:
        client = self._get_client()
        context = self._build_decision_context(question, visible_log)
        context["question"] = question
        self._write_journal("btw_request_context", context)
        answer = client.ask_btw(context)
        self._write_journal("btw_response", {"answer": answer})
        self.log_line(answer, kind="agent")

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
            self.log_line(line.rstrip("\n"), kind="program", event="stdout")

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self.log_line(line.rstrip("\n"), kind="error", event="stderr")

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
            self.log_line(f"bad frame event: {exc}", kind="error", event="bad_frame_event")
            return
        self.recent_events.append(event)
        if self.log.log_level == "DEBUG" and event.filename.endswith(self.script_path.name):
            self.log_line(
                f"{event.event} {Path(event.filename).name}:{event.lineno} {event.function}",
                kind="trace",
                event="trace_display",
            )

    def _wait_for_exit(self) -> None:
        assert self.process is not None
        self.returncode = self.process.wait()
        with self._state_lock:
            self.finished = True
            self.paused = False
        if self.paths.pause_path.exists():
            self.paths.pause_path.unlink()
        self.log_line(f"native process exited rc={self.returncode}", kind="system", event="process_exit")

    def log_line(
        self,
        line: str,
        *,
        kind: str = "system",
        label: str | None = None,
        event: str = "tui_log",
    ) -> None:
        self.log.append(line, kind=kind, label=label)
        self._write_journal(event, {"kind": kind, "label": label, "text": line})

    def _write_journal(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            "run_id": self.run_id,
            **payload,
        }
        with self._journal_lock:
            self.paths.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.paths.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _build_decision_context(self, instruction: str, visible_log: list[str]) -> dict[str, Any]:
        recent_events = list(self.recent_events)[-40:]
        return {
            "user_instruction": instruction,
            "script": str(self.script_path),
            "script_symbols": _script_symbols(self.script_path),
            "recent_log": visible_log[-80:],
            "recent_frame_events": [_event_summary(event) for event in recent_events],
            "recent_source_windows": _source_windows(self.script_path, recent_events),
            "runtime_notes": [
                "Injected code runs at the next CPython trace safepoint with the current frame globals and locals.",
                "Use only names visible in script_symbols/source windows, or guard lookups with globals().get/locals().get.",
                "For inspection requests, print a useful fallback message instead of raising if a variable is absent.",
                "For the native MNIST example, batch state is exposed through last_batch_summary and batch_loss_trace.",
            ],
        }


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
    session.log_line("Native CPython TUI. Commands: /pause /resume /exec <code> /btw <question> /quit.", kind="system")

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
        session.log_line("Use /exec <python code> for direct CPython-frame execution.", kind="system")
        session.log_line("Natural-language text pauses, asks Codex for Python code, queues it, and waits for /resume.", kind="system")
        return False
    if normalized == "/btw" or normalized.startswith("/btw "):
        question = text[4:].strip()
        if not question:
            session.log_line("Usage: /btw <side question>", kind="system")
            return False
        session.log_line(question, kind="btw", event="btw_input")
        session.submit_btw(question, [line.text for line in log.render_lines(width=120, max_lines=80)])
        return False
    if normalized == "/exec" or normalized.startswith("/exec "):
        code = text[6:].strip()
        if not code:
            session.log_line("Usage: /exec <python code>", kind="system")
            return False
        session.pause()
        session.queue_code(code, source="operator /exec")
        session.log_line("direct code queued; still paused until /resume", kind="agent")
        return False

    session.log_line(text, kind="user", event="user_input")
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


def _source_windows(script_path: Path, events: list[FrameEvent], *, radius: int = 3) -> list[dict[str, Any]]:
    try:
        lines = script_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    windows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for event in reversed(events):
        if Path(event.filename).resolve() != script_path.resolve():
            continue
        key = (event.function, event.lineno)
        if key in seen:
            continue
        seen.add(key)
        start = max(1, event.lineno - radius)
        end = min(len(lines), event.lineno + radius)
        windows.append(
            {
                "function": event.function,
                "current_lineno": event.lineno,
                "lines": [
                    {
                        "lineno": lineno,
                        "source": lines[lineno - 1],
                    }
                    for lineno in range(start, end + 1)
                ],
            }
        )
        if len(windows) >= 8:
            break
    return list(reversed(windows))


def _script_symbols(script_path: Path) -> dict[str, list[str]]:
    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    except (OSError, SyntaxError):
        return {"assigned": [], "functions": [], "classes": [], "imports": []}

    assigned: set[str] = set()
    functions: set[str] = set()
    classes: set[str] = set()
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
            assigned.add(node.name)
            continue
        if isinstance(node, ast.ClassDef):
            classes.add(node.name)
            assigned.add(node.name)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name.split(".")[0])
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
            continue
        for target in _top_level_targets(node):
            assigned.add(target)
    return {
        "assigned": sorted(assigned),
        "functions": sorted(functions),
        "classes": sorted(classes),
        "imports": sorted(imports),
    }


def _top_level_targets(node: ast.AST) -> list[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    elif isinstance(node, ast.AugAssign):
        targets.append(node.target)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        targets.append(node.target)
        for child in node.body:
            targets.extend(_assignment_targets_in_statement(child))
    elif isinstance(node, ast.With):
        for item in node.items:
            if item.optional_vars is not None:
                targets.append(item.optional_vars)
    return [name for target in targets for name in _target_names(target)]


def _assignment_targets_in_statement(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    if isinstance(node, ast.AugAssign):
        return [node.target]
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return [node.target]
    return []


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _target_names(element)]
    return []
