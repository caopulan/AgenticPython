import json
from pathlib import Path

import pytest

from agenticpython.native.protocol import FrameEvent
from agenticpython.native.tui import (
    _handle_native_input,
    _parse_natural_language_control,
    _resolve_package_filter,
    _script_symbols,
    _source_windows,
    build_native_env,
    make_native_run_paths,
    NativeProcessSession,
    resolve_native_python,
    write_control_command,
    write_exec_command,
)
from agenticpython.tui import TuiLog


def test_make_native_run_paths_uses_out_dir(tmp_path):
    paths = make_native_run_paths(tmp_path / "run")

    assert paths.events_path == tmp_path / "run" / "frame_events.jsonl"
    assert paths.commands_path == tmp_path / "run" / "commands.jsonl"
    assert paths.journal_path == tmp_path / "run" / "journal.jsonl"
    assert paths.pause_path == tmp_path / "run" / "PAUSED"
    assert paths.command_dir == tmp_path / "run" / "commands"


def test_build_native_env_sets_cpython_probe_variables(tmp_path):
    paths = make_native_run_paths(tmp_path / "run")
    script_path = tmp_path / "examples" / "native_cpu_mnist.py"
    env = build_native_env(
        {"PYTHONPATH": "existing"},
        repo_root=tmp_path,
        paths=paths,
        run_id="run-1",
        script_path=script_path,
    )

    assert env["PYTHON_AGENTIC"] == "1"
    assert env["PYTHON_AGENTIC_RUN_ID"] == "run-1"
    assert env["PYTHON_AGENTIC_EVENTS"] == str(paths.events_path)
    assert env["PYTHON_AGENTIC_COMMANDS"] == str(paths.commands_path)
    assert env["PYTHON_AGENTIC_PAUSE_FILE"] == str(paths.pause_path)
    assert env["PYTHON_AGENTIC_FILTER"] == str(script_path)
    assert env["PYTHONUNBUFFERED"] == "1"
    assert str(tmp_path / "src") in env["PYTHONPATH"]
    assert "existing" in env["PYTHONPATH"]


def test_write_exec_command_appends_code_file_path(tmp_path):
    commands_path = tmp_path / "commands.jsonl"
    code_path = tmp_path / "commands" / "command-0001.py"

    write_exec_command(commands_path, code_path)

    assert commands_path.read_text(encoding="utf-8") == f"exec_file\t{code_path}\n"


def test_write_control_command_appends_tab_separated_command(tmp_path):
    commands_path = tmp_path / "commands.jsonl"

    write_control_command(commands_path, "set_filters", "script.py", "torch/optim")

    assert commands_path.read_text(encoding="utf-8") == "set_filters\tscript.py\ttorch/optim\n"


def test_resolve_package_filter_returns_package_directory():
    resolved = _resolve_package_filter("torch.optim")

    assert resolved is not None
    assert resolved.endswith("torch/optim")


def test_parse_natural_language_control_stops_in_optimizer_calls():
    request = _parse_natural_language_control("我要在 optim 里停")

    assert request is not None
    assert request.trace_package == "torch.optim"
    assert request.break_mode == "call"
    assert request.resume is True


def test_parse_natural_language_control_stops_on_optimizer_lines():
    request = _parse_natural_language_control("我想在 optimizer 每一行停住")

    assert request is not None
    assert request.trace_package == "torch.optim"
    assert request.break_mode == "line"
    assert request.resume is True


def test_parse_natural_language_control_can_trace_all_python_lines():
    request = _parse_natural_language_control("在任意 python 代码每一行都停")

    assert request is not None
    assert request.trace_all is True
    assert request.break_mode == "line"
    assert request.resume is True


def test_parse_natural_language_control_ignores_regular_questions():
    assert _parse_natural_language_control("现在多少iter了") is None


def test_parse_natural_language_control_can_turn_breaks_off():
    request = _parse_natural_language_control("optim 里不要停了")

    assert request is not None
    assert request.break_mode == "off"
    assert request.resume is True


def test_handle_native_input_applies_natural_language_optimizer_break(tmp_path):
    log = TuiLog()
    session = NativeProcessSession(
        script_path=Path("examples/native_cpu_mnist.py"),
        out_dir=tmp_path / "run",
        native_python=tmp_path / "python",
        repo_root=Path.cwd(),
        log=log,
    )
    session.paths.out_dir.mkdir(parents=True, exist_ok=True)
    session.paths.commands_path.write_text("", encoding="utf-8")

    should_quit = _handle_native_input("我要在 optim 里停", session, log)

    command_lines = session.paths.commands_path.read_text(encoding="utf-8").splitlines()
    assert should_quit is False
    assert any(line.startswith("add_filter\t") and line.endswith("torch/optim") for line in command_lines)
    assert "set_break_mode\tcall" in command_lines
    assert "set_step_mode\tnone" in command_lines
    assert "resume" in command_lines
    assert any(entry.text == "natural-language runtime control applied" for entry in log.snapshot())


def test_resolve_native_python_uses_requested_path(tmp_path):
    python_path = tmp_path / "python"
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")

    assert resolve_native_python(tmp_path, python_path) == python_path


def test_resolve_native_python_errors_when_missing(tmp_path):
    with pytest.raises(RuntimeError, match="Patched CPython runtime is missing"):
        resolve_native_python(tmp_path, None)


def test_script_symbols_include_native_mnist_training_state():
    symbols = _script_symbols(Path("examples/native_cpu_mnist.py"))

    assert "num_epochs" in symbols["assigned"]
    assert "batch_index" in symbols["assigned"]
    assert "total_batches" in symbols["assigned"]
    assert "loss_value" in symbols["assigned"]
    assert "summarize_current_grads" in symbols["functions"]


def test_source_windows_include_recent_frame_source():
    script_path = Path("examples/native_cpu_mnist.py").resolve()
    loop_lineno = next(
        index
        for index, line in enumerate(script_path.read_text(encoding="utf-8").splitlines(), start=1)
        if "for epoch, batch_index, total_batches" in line
    )
    windows = _source_windows(
        script_path,
        [
            FrameEvent(
                event="line",
                run_id="run",
                process_id=1,
                frame_id="frame",
                filename=str(script_path),
                function="<module>",
                lineno=loop_lineno,
            )
        ],
    )

    flattened = "\n".join(line["source"] for window in windows for line in window["lines"])
    assert "for epoch, batch_index, total_batches" in flattened
    assert "loss_value, batch_examples" in flattened


def test_native_session_log_line_writes_journal(tmp_path):
    session = NativeProcessSession(
        script_path=Path("examples/native_cpu_mnist.py"),
        out_dir=tmp_path / "run",
        native_python=tmp_path / "python",
        repo_root=Path.cwd(),
        log=TuiLog(),
    )

    session.log_line("hello", kind="system", event="test_event")

    records = [
        json.loads(line)
        for line in session.paths.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["event"] == "test_event"
    assert records[-1]["text"] == "hello"
