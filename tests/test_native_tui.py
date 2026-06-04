from pathlib import Path

import pytest

from agenticpython.native.tui import (
    build_native_env,
    make_native_run_paths,
    resolve_native_python,
    write_exec_command,
)


def test_make_native_run_paths_uses_out_dir(tmp_path):
    paths = make_native_run_paths(tmp_path / "run")

    assert paths.events_path == tmp_path / "run" / "frame_events.jsonl"
    assert paths.commands_path == tmp_path / "run" / "commands.jsonl"
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
    assert str(tmp_path / "src") in env["PYTHONPATH"]
    assert "existing" in env["PYTHONPATH"]


def test_write_exec_command_appends_code_file_path(tmp_path):
    commands_path = tmp_path / "commands.jsonl"
    code_path = tmp_path / "commands" / "command-0001.py"

    write_exec_command(commands_path, code_path)

    assert commands_path.read_text(encoding="utf-8") == f"exec_file\t{code_path}\n"


def test_resolve_native_python_uses_requested_path(tmp_path):
    python_path = tmp_path / "python"
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")

    assert resolve_native_python(tmp_path, python_path) == python_path


def test_resolve_native_python_errors_when_missing(tmp_path):
    with pytest.raises(RuntimeError, match="Patched CPython runtime is missing"):
        resolve_native_python(tmp_path, None)
