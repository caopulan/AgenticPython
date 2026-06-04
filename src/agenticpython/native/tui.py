from __future__ import annotations

import ast
from collections import deque
import curses
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
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


@dataclass(frozen=True)
class NaturalLanguageControlRequest:
    trace_package: str | None = None
    trace_all: bool = False
    trace_script: bool = False
    break_mode: str | None = None
    step_mode: str | None = None
    resume: bool = False


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


def write_control_command(commands_path: Path, command: str, *args: str) -> None:
    with commands_path.open("a", encoding="utf-8") as handle:
        fields = [command, *args]
        handle.write("\t".join(fields) + "\n")


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
        self._trace_filters: list[str] = [str(self.script_path)]
        self._break_mode = "off"
        self._step_mode = "none"
        self._state_lock = threading.RLock()
        self._journal_lock = threading.RLock()

    def start(self) -> None:
        if self.paths.command_dir.exists():
            shutil.rmtree(self.paths.command_dir)
        self.paths.command_dir.mkdir(parents=True, exist_ok=True)
        self._command_index = 0
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
                "trace_filters": self._trace_filters,
                "break_mode": self._break_mode,
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
        write_control_command(self.paths.commands_path, "resume")
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

    def set_trace_filters(self, filters: list[str], *, source: str = "operator") -> None:
        normalized = [item for item in filters if item]
        with self._state_lock:
            self._trace_filters = normalized
        if normalized:
            write_control_command(self.paths.commands_path, "set_filters", *normalized)
            rendered = ", ".join(normalized)
        else:
            write_control_command(self.paths.commands_path, "clear_filters")
            rendered = "all Python frames"
        self._write_journal("trace_filters_set", {"source": source, "filters": normalized})
        self.log_line(f"trace filters: {rendered}", kind="system", event="trace_filters_set")

    def add_trace_filter(self, filter_value: str, *, source: str = "operator") -> None:
        normalized = filter_value
        with self._state_lock:
            if normalized not in self._trace_filters:
                self._trace_filters.append(normalized)
        write_control_command(self.paths.commands_path, "add_filter", normalized)
        self._write_journal("trace_filter_added", {"source": source, "filter": normalized})
        self.log_line(f"trace filter added: {normalized}", kind="system", event="trace_filter_added")

    def set_break_mode(self, mode: str) -> None:
        allowed = {"off", "line", "call", "return", "exception", "all"}
        if mode not in allowed:
            self.log_line(f"unknown break mode: {mode}", kind="error")
            return
        with self._state_lock:
            self._break_mode = mode
        write_control_command(self.paths.commands_path, "set_break_mode", mode)
        self._write_journal("break_mode_set", {"mode": mode})
        self.log_line(f"break mode: {mode}", kind="system", event="break_mode_set")

    def step(self, mode: str) -> None:
        allowed = {"into", "over", "out"}
        if mode not in allowed:
            self.log_line(f"unknown step mode: {mode}", kind="error")
            return
        with self._state_lock:
            self._step_mode = mode
        write_control_command(self.paths.commands_path, "set_step_mode", mode)
        self.resume()
        self._write_journal("step_mode_set", {"mode": mode})
        self.log_line(f"step mode: {mode}", kind="system", event="step_mode_set")

    def continue_execution(self) -> None:
        with self._state_lock:
            self._step_mode = "none"
        write_control_command(self.paths.commands_path, "set_step_mode", "none")
        self.resume()

    def trace_status(self) -> str:
        with self._state_lock:
            filters = list(self._trace_filters)
            break_mode = self._break_mode
            step_mode = self._step_mode
        filters_text = ", ".join(filters) if filters else "all Python frames"
        return f"trace={filters_text}; break={break_mode}; step={step_mode}"

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
                "Functions created by injected code run later with globals, so bind local state with default arguments or store it in globals/runtime objects.",
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
    session.log_line(
        "Native CPython TUI. Commands: /pause /resume /continue /trace /break /step /next /out /exec /btw /quit.",
        kind="system",
    )

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
    if normalized in {"/continue", "/cont", "/c", "continue", "cont"}:
        session.continue_execution()
        return False
    if normalized in {"/step", "/s", "step"}:
        session.step("into")
        return False
    if normalized in {"/next", "/n", "next"}:
        session.step("over")
        return False
    if normalized in {"/out", "/finish", "out", "finish"}:
        session.step("out")
        return False
    if normalized in {"/help", "help", "帮助"}:
        session.log_line("Use /exec <python code> for direct CPython-frame execution.", kind="system")
        session.log_line("Use /trace script|all|package <name>|path <path>|show to control trace scope.", kind="system")
        session.log_line("Use /break off|line|call|return|exception|all and /step /next /out /continue to control depth.", kind="system")
        session.log_line("Natural-language controls such as '我要在 optim 里停' adjust trace/break state locally.", kind="system")
        session.log_line("Other natural-language text pauses, asks Codex for Python code, queues it, and waits for /resume.", kind="system")
        return False
    if normalized == "/trace" or normalized.startswith("/trace "):
        _handle_trace_command(text, session)
        return False
    if normalized == "/break" or normalized.startswith("/break "):
        _handle_break_command(text, session)
        return False
    natural_control = _parse_natural_language_control(text)
    if natural_control is not None:
        session.log_line(text, kind="user", event="user_input")
        _apply_natural_language_control(natural_control, session)
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


def _handle_trace_command(text: str, session: NativeProcessSession) -> None:
    parts = text.split(maxsplit=2)
    if len(parts) == 1 or parts[1] == "show":
        session.log_line(session.trace_status(), kind="system", event="trace_status")
        return
    mode = parts[1]
    value = parts[2].strip() if len(parts) > 2 else ""
    if mode == "script":
        session.set_trace_filters([str(session.script_path)], source="trace script")
        return
    if mode == "all":
        session.set_trace_filters([], source="trace all")
        return
    if mode == "clear":
        session.set_trace_filters([str(session.script_path)], source="trace clear")
        return
    if mode == "package":
        if not value:
            session.log_line("Usage: /trace package <module.name>", kind="system")
            return
        resolved = _resolve_package_filter(value)
        if resolved is None:
            session.log_line(f"could not resolve package: {value}", kind="error")
            return
        session.add_trace_filter(resolved, source=f"trace package {value}")
        return
    if mode == "path":
        if not value:
            session.log_line("Usage: /trace path <path-substring>", kind="system")
            return
        session.add_trace_filter(value, source="trace path")
        return
    if mode == "set":
        if not value:
            session.log_line("Usage: /trace set <path-substring>", kind="system")
            return
        session.set_trace_filters([value], source="trace set")
        return
    session.log_line("Usage: /trace script|all|clear|package <name>|path <path>|set <path>|show", kind="system")


def _handle_break_command(text: str, session: NativeProcessSession) -> None:
    parts = text.split(maxsplit=1)
    if len(parts) == 1:
        session.log_line("Usage: /break off|line|call|return|exception|all", kind="system")
        return
    session.set_break_mode(parts[1].strip().lower())


def _parse_natural_language_control(text: str) -> NaturalLanguageControlRequest | None:
    normalized = text.strip().lower().replace(" ", "")
    if not normalized:
        return None

    mentions_python_all = (
        "任意python" in normalized
        or "所有python" in normalized
        or "全部python" in normalized
        or "allpython" in normalized
        or "任何python" in normalized
    )
    mentions_optimizer = (
        "torch.optim" in normalized
        or "optimizer" in normalized
        or "optim" in normalized
        or "优化器" in normalized
        or "梯度更新" in normalized
        or "更新参数" in normalized
    )

    wants_break_off = any(keyword in normalized for keyword in ["不停了", "别停", "不要停", "不用停", "关闭break"])
    wants_stop = any(keyword in normalized for keyword in ["停", "断", "break", "stop"])
    wants_stop = wants_stop or ((mentions_optimizer or mentions_python_all) and any(keyword in normalized for keyword in ["听一下", "听下"]))
    wants_step = any(keyword in normalized for keyword in ["step", "单步", "一步", "踩进去"])
    wants_continue = any(keyword in normalized for keyword in ["继续", "resume", "continue"])

    if wants_break_off:
        return NaturalLanguageControlRequest(break_mode="off", resume=True)
    if wants_continue and not wants_stop and not wants_step:
        return NaturalLanguageControlRequest(break_mode="off", resume=True)
    if not wants_stop and not wants_step:
        return None
    if not mentions_optimizer and not mentions_python_all:
        return None

    break_mode = "call"
    if any(keyword in normalized for keyword in ["每一行", "每行", "逐行", "line", "行级"]):
        break_mode = "line"
    elif any(keyword in normalized for keyword in ["异常", "exception"]):
        break_mode = "exception"
    elif any(keyword in normalized for keyword in ["返回", "return"]):
        break_mode = "return"
    elif any(keyword in normalized for keyword in ["所有事件", "all"]):
        break_mode = "all"

    step_mode = "into" if wants_step and "next" not in normalized else None
    return NaturalLanguageControlRequest(
        trace_package="torch.optim" if mentions_optimizer else None,
        trace_all=mentions_python_all,
        break_mode=break_mode,
        step_mode=step_mode,
        resume=True,
    )


def _apply_natural_language_control(request: NaturalLanguageControlRequest, session: NativeProcessSession) -> None:
    if request.trace_all:
        session.set_trace_filters([], source="natural language")
    if request.trace_script:
        session.set_trace_filters([str(session.script_path)], source="natural language")
    if request.trace_package is not None:
        resolved = _resolve_package_filter(request.trace_package)
        if resolved is None:
            session.log_line(f"could not resolve package: {request.trace_package}", kind="error")
            return
        session.add_trace_filter(resolved, source=f"natural language package {request.trace_package}")
    if request.break_mode is not None:
        session.set_break_mode(request.break_mode)
    if request.step_mode is not None:
        session.step(request.step_mode)
        return
    if request.resume:
        session.continue_execution()
    session.log_line("natural-language runtime control applied", kind="agent", event="natural_control_applied")


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
        f" {status.upper()}  {session.trace_status()}  out={session.out_dir}",
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


def _resolve_package_filter(package_name: str) -> str | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return str(Path(next(iter(spec.submodule_search_locations))).resolve())
    if spec.origin is not None:
        return str(Path(spec.origin).resolve())
    return None
