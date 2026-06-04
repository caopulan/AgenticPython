import time

from agenticpython.actions import ScriptedActionClient
from agenticpython.interactive import BackgroundTicker, InteractiveSession
from agenticpython.runtime import AgenticRunner
from agenticpython.tui import TuiLog


def test_user_instruction_pauses_applies_patch_and_waits_for_resume(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("x = 1\nx = x + 1\nprint(x)\n", encoding="utf-8")
    client = ScriptedActionClient(
        {
            "human": {
                "operations": [
                    {"op": "insert_before", "target": "__current__", "code": "x = 10"}
                ],
                "resume": True,
            }
        }
    )
    runner = AgenticRunner.from_path(script, decision_client=client, out_dir=tmp_path / "run")
    session = InteractiveSession(runner)

    session.tick()
    session.submit_instruction("change x before the next instruction")
    session.tick()

    assert session.paused is True
    assert runner.namespace["x"] == 1
    assert client.calls[0]["user_instruction"] == "change x before the next instruction"

    session.resume()
    while not session.finished:
        session.tick()

    assert runner.namespace["x"] == 11


def test_pause_and_resume_do_not_call_decision_client(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("x = 1\nx = x + 1\n", encoding="utf-8")
    client = ScriptedActionClient({})
    runner = AgenticRunner.from_path(script, decision_client=client, out_dir=tmp_path / "run")
    session = InteractiveSession(runner)

    session.pause()
    session.tick()
    assert runner.namespace.get("x") is None

    session.resume()
    while not session.finished:
        session.tick()

    assert runner.namespace["x"] == 2
    assert client.calls == []


def test_background_ticker_keeps_control_thread_responsive_during_long_instruction(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("import time\ntime.sleep(0.2)\nx = 1\n", encoding="utf-8")
    runner = AgenticRunner.from_path(
        script,
        decision_client=ScriptedActionClient({}),
        out_dir=tmp_path / "run",
    )
    session = InteractiveSession(runner)
    ticker = BackgroundTicker(session, delay=0.001)

    ticker.start()
    time.sleep(0.05)
    started_at = time.perf_counter()
    session.pause()
    elapsed = time.perf_counter() - started_at
    ticker.stop()

    assert elapsed < 0.05
    assert session.paused is True
    assert "x" not in runner.namespace


def test_instruction_submitted_during_running_step_applies_to_next_instruction(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("import time\ntime.sleep(0.1)\nx = 1\n", encoding="utf-8")
    client = ScriptedActionClient(
        {
            "human": {
                "operations": [
                    {"op": "replace", "target": "__current__", "code": "x = 10"}
                ],
                "resume": True,
            }
        }
    )
    runner = AgenticRunner.from_path(script, decision_client=client, out_dir=tmp_path / "run")
    session = InteractiveSession(runner)
    ticker = BackgroundTicker(session, delay=0.001)

    ticker.start()
    time.sleep(0.03)
    session.submit_instruction("replace the next instruction with x = 10")
    time.sleep(0.2)

    assert session.paused is True
    assert runner.namespace.get("x") is None
    assert client.calls[0]["current_instruction"]["source"] == "x = 1"

    session.resume()
    deadline = time.monotonic() + 1
    while not session.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    ticker.stop()

    assert runner.namespace["x"] == 10


def test_queued_instruction_is_applied_by_background_ticker(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("x = 1\n", encoding="utf-8")
    client = ScriptedActionClient(
        {
            "human": {
                "operations": [
                    {"op": "replace", "target": "__current__", "code": "x = 10"}
                ],
                "resume": True,
            }
        }
    )
    runner = AgenticRunner.from_path(script, decision_client=client, out_dir=tmp_path / "run")
    session = InteractiveSession(runner)

    session.queue_instruction("replace x with 10")

    assert client.calls == []

    ticker = BackgroundTicker(session, delay=0.001)
    ticker.start()
    deadline = time.monotonic() + 1
    while not client.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    ticker.stop()

    assert client.calls[0]["current_instruction"]["source"] == "x = 1"
    assert session.paused is True


def test_runner_can_send_stdout_to_event_sink_without_echoing(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("print('hello tui')\n", encoding="utf-8")
    events = []
    runner = AgenticRunner.from_path(
        script,
        decision_client=ScriptedActionClient({}),
        out_dir=tmp_path / "run",
        echo_stdout=False,
        event_sink=events.append,
    )

    runner.run()

    assert any(event.get("stdout") == "hello tui\n" for event in events)


def test_runner_captures_logging_configured_in_earlier_instruction(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text(
        "import logging\n"
        "import sys\n"
        "logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s', force=True)\n"
        "logging.info('hello logging')\n",
        encoding="utf-8",
    )
    events = []
    runner = AgenticRunner.from_path(
        script,
        decision_client=ScriptedActionClient({}),
        out_dir=tmp_path / "run",
        echo_stdout=False,
        event_sink=events.append,
    )

    runner.run()

    assert any(event.get("stdout") == "hello logging\n" for event in events)


def test_stop_before_first_tick_still_writes_artifacts(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("x = 1\n", encoding="utf-8")
    runner = AgenticRunner.from_path(
        script,
        decision_client=ScriptedActionClient({}),
        out_dir=tmp_path / "run",
    )
    session = InteractiveSession(runner)

    session.stop()

    assert (tmp_path / "run" / "final_tape.json").exists()
    assert (tmp_path / "run" / "replay.py").exists()


def test_tui_log_hides_instruction_trace_at_info_and_shows_stdout():
    log = TuiLog(log_level="INFO")

    log.event_sink(
        {
            "event": "execute",
            "instruction": {"id": "I0001", "source": "x = 1"},
            "status": "ok",
            "stdout": "epoch=1 accuracy=0.8\n",
        }
    )

    assert list(log.lines) == ["epoch=1 accuracy=0.8"]


def test_tui_log_shows_instruction_trace_at_debug():
    log = TuiLog(log_level="DEBUG")

    log.event_sink(
        {
            "event": "execute",
            "instruction": {"id": "I0001", "source": "x = 1"},
            "status": "ok",
            "stdout": "",
        }
    )

    assert list(log.lines) == ["[ok] I0001: x = 1"]
