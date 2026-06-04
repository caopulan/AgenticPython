import os
from pathlib import Path
import selectors
import subprocess
import time

import pytest

from agenticpython.native.tui import (
    build_native_env,
    make_native_run_paths,
    resolve_native_python,
    write_control_command,
    write_exec_command,
)


def test_native_command_channel_executes_queued_code(tmp_path):
    repo_root = Path.cwd()
    try:
        native_python = resolve_native_python(repo_root, None)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    target = tmp_path / "native_lr_target.py"
    target.write_text(
        "\n".join(
            [
                "import time",
                "lr = 0.1",
                "for step in range(8):",
                "    print(f'step={step} lr={lr}', flush=True)",
                "    time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths = make_native_run_paths(tmp_path / "run")
    paths.out_dir.mkdir(parents=True)
    paths.command_dir.mkdir()
    paths.events_path.write_text("", encoding="utf-8")
    paths.commands_path.write_text("", encoding="utf-8")
    env = build_native_env(
        os.environ,
        repo_root=repo_root,
        paths=paths,
        run_id="native-command-test",
        script_path=target,
    )

    process = subprocess.Popen(
        [str(native_python), str(target)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    seen = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            assert process.stdout is not None
            line = process.stdout.readline()
            if not line:
                break
            seen.append(line)
            if "step=1 lr=0.1" in line:
                break

        command = paths.command_dir / "set_lr.py"
        command.write_text(
            "lr = 0.01\nprint(f'agent changed lr={lr}', flush=True)\n",
            encoding="utf-8",
        )
        write_exec_command(paths.commands_path, command)

        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)

    output = "".join(seen) + stdout
    assert process.returncode == 0, stderr
    assert "agent changed lr=0.01" in output
    assert "lr=0.01" in output


def test_native_command_prints_while_process_is_paused(tmp_path):
    repo_root = Path.cwd()
    try:
        native_python = resolve_native_python(repo_root, None)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    target = tmp_path / "paused_target.py"
    target.write_text(
        "\n".join(
            [
                "import time",
                "value = 42",
                "for step in range(100):",
                "    marker = step",
                "    time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths = make_native_run_paths(tmp_path / "run")
    paths.out_dir.mkdir(parents=True)
    paths.command_dir.mkdir()
    paths.events_path.write_text("", encoding="utf-8")
    paths.commands_path.write_text("", encoding="utf-8")
    paths.pause_path.write_text("paused\n", encoding="utf-8")
    env = build_native_env(
        os.environ,
        repo_root=repo_root,
        paths=paths,
        run_id="native-paused-command-test",
        script_path=target,
    )

    process = subprocess.Popen(
        [str(native_python), str(target)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not paths.events_path.read_text(encoding="utf-8").strip():
            time.sleep(0.05)

        command = paths.command_dir / "inspect.py"
        command.write_text(
            "print(f'paused command ran value={globals().get(\"value\")}', flush=False)\n",
            encoding="utf-8",
        )
        write_exec_command(paths.commands_path, command)

        output = ""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            for key, _ in selector.select(timeout=0.1):
                output += key.fileobj.readline()
            if "paused command ran" in output:
                break
        assert "paused command ran" in output
    finally:
        if paths.pause_path.exists():
            paths.pause_path.unlink()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)


def test_native_trace_filter_can_expand_to_imported_python_file(tmp_path):
    repo_root = Path.cwd()
    try:
        native_python = resolve_native_python(repo_root, None)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    helper = tmp_path / "helper.py"
    helper.write_text(
        "\n".join(
            [
                "def compute(value):",
                "    total = value + 1",
                "    return total",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "trace_target.py"
    target.write_text(
        "\n".join(
            [
                "import time",
                "import helper",
                "for step in range(30):",
                "    helper.compute(step)",
                "    time.sleep(0.05)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths = make_native_run_paths(tmp_path / "run")
    paths.out_dir.mkdir(parents=True)
    paths.command_dir.mkdir()
    paths.events_path.write_text("", encoding="utf-8")
    paths.commands_path.write_text("", encoding="utf-8")
    env = build_native_env(
        {**os.environ, "PYTHONPATH": str(tmp_path)},
        repo_root=repo_root,
        paths=paths,
        run_id="native-dynamic-filter-test",
        script_path=target,
    )

    process = subprocess.Popen(
        [str(native_python), str(target)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "trace_target.py" not in paths.events_path.read_text(encoding="utf-8"):
            time.sleep(0.05)
        write_control_command(paths.commands_path, "set_filters", str(target), str(helper))
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)

    events = paths.events_path.read_text(encoding="utf-8")
    assert process.returncode == 0
    assert "trace_target.py" in events
    assert "helper.py" in events


def test_native_break_mode_stops_until_resumed(tmp_path):
    repo_root = Path.cwd()
    try:
        native_python = resolve_native_python(repo_root, None)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    target = tmp_path / "break_target.py"
    target.write_text(
        "\n".join(
            [
                "import time",
                "for step in range(80):",
                "    marker = step",
                "    time.sleep(0.02)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths = make_native_run_paths(tmp_path / "run")
    paths.out_dir.mkdir(parents=True)
    paths.command_dir.mkdir()
    paths.events_path.write_text("", encoding="utf-8")
    paths.commands_path.write_text("", encoding="utf-8")
    env = build_native_env(
        os.environ,
        repo_root=repo_root,
        paths=paths,
        run_id="native-break-mode-test",
        script_path=target,
    )

    process = subprocess.Popen(
        [str(native_python), str(target)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "break_target.py" not in paths.events_path.read_text(encoding="utf-8"):
            time.sleep(0.05)
        write_control_command(paths.commands_path, "set_break_mode", "line")
        time.sleep(0.3)
        assert process.poll() is None
        write_control_command(paths.commands_path, "set_break_mode", "off")
        write_control_command(paths.commands_path, "resume")
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)

    assert process.returncode == 0
