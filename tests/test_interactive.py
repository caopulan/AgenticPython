import time

from agenticpython.actions import ScriptedActionClient
from agenticpython.interactive import BackgroundTicker, InteractiveSession
from agenticpython.runtime import AgenticRunner
from agenticpython.tui import TuiLog, _KIND_COLOR_PAIRS, _display_width, _handle_input, _prompt_view


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


def test_btw_question_does_not_pause_or_patch_runtime(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("x = 1\n", encoding="utf-8")
    client = ScriptedActionClient({}, btw_responses=["loss is still noisy"])
    runner = AgenticRunner.from_path(script, decision_client=client, out_dir=tmp_path / "run")
    session = InteractiveSession(runner)

    session.submit_btw("how is the loss?", visible_log=["OUT epoch=1 batch=10 loss=2.3"])
    session.wait_for_btw(timeout=1)

    assert session.paused is False
    assert runner.namespace.get("x") is None
    assert client.calls == []
    assert client.btw_calls[0]["question"] == "how is the loss?"
    assert client.btw_calls[0]["visible_log"] == ["OUT epoch=1 batch=10 loss=2.3"]
    assert "btw answer: loss is still noisy" in session.drain_messages()


def test_btw_question_can_run_while_current_instruction_is_busy(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("import time\ntime.sleep(0.2)\nx = 1\n", encoding="utf-8")
    client = ScriptedActionClient({}, btw_responses=["training continues"])
    runner = AgenticRunner.from_path(script, decision_client=client, out_dir=tmp_path / "run")
    session = InteractiveSession(runner)
    ticker = BackgroundTicker(session, delay=0.001)

    ticker.start()
    time.sleep(0.03)
    started_at = time.perf_counter()
    session.submit_btw("are you alive?")
    elapsed = time.perf_counter() - started_at
    session.wait_for_btw(timeout=1)
    ticker.stop()

    assert elapsed < 0.05
    assert session.paused is False
    assert client.btw_calls[0]["runner_busy"] is True
    assert "x" not in runner.namespace


def test_tui_btw_command_keeps_session_running(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text("x = 1\n", encoding="utf-8")
    client = ScriptedActionClient({}, btw_responses=["I can answer while running"])
    runner = AgenticRunner.from_path(script, decision_client=client, out_dir=tmp_path / "run")
    session = InteractiveSession(runner)
    log = TuiLog(log_level="INFO")

    should_quit = _handle_input("/btw 当前loss怎么样", session, log)
    session.wait_for_btw(timeout=1)
    for message in session.drain_messages():
        log.append(message, kind="agent")

    rendered = log.render_lines(width=80, max_lines=10)

    assert should_quit is False
    assert session.paused is False
    assert client.calls == []
    assert client.btw_calls[0]["question"] == "当前loss怎么样"
    assert any(line.text == "BTW       当前loss怎么样" for line in rendered)
    assert any("I can answer while running" in line.text for line in rendered)


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


def test_execute_now_stdout_is_routed_to_event_sink_without_echoing(tmp_path, capsys):
    script = tmp_path / "demo.py"
    script.write_text("x = 1\n", encoding="utf-8")
    events = []
    client = ScriptedActionClient(
        {
            "human": {
                "operations": [
                    {"op": "execute_now", "code": "print('AI 回答应该进上方 log')"}
                ],
                "resume": True,
            }
        }
    )
    runner = AgenticRunner.from_path(
        script,
        decision_client=client,
        out_dir=tmp_path / "run",
        echo_stdout=False,
        event_sink=events.append,
    )
    session = InteractiveSession(runner)

    session.submit_instruction("print a reply")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert any(
        event.get("event") == "execute_now" and event.get("stdout") == "AI 回答应该进上方 log\n"
        for event in events
    )


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

    rendered = log.render_lines(width=80, max_lines=10)

    assert [line.text for line in rendered] == ["OUT       epoch=1 accuracy=0.8"]


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

    rendered = log.render_lines(width=80, max_lines=10)

    assert [line.text for line in rendered] == ["TRACE     ok I0001: x = 1"]


def test_tui_color_mapping_uses_blue_for_codex_and_black_for_trace():
    assert _KIND_COLOR_PAIRS["trigger"] == 3
    assert _KIND_COLOR_PAIRS["trace"] == 9


def test_tui_log_collapses_patch_code_and_wraps_to_width():
    log = TuiLog(log_level="INFO")

    log.event_sink(
        {
            "event": "patch",
            "action": {
                "operations": [
                    {
                        "op": "insert_before",
                        "target": "__current__",
                        "code": (
                            'if "batch_loss_trace" not in globals():\n'
                            "    batch_loss_trace = []\n"
                            "batch_loss_trace.append((int(epoch), int(batch_index), float(loss_value)))"
                        ),
                    }
                ]
            },
        }
    )

    rendered = log.render_lines(width=56, max_lines=10)

    assert rendered[0].text.startswith("PATCH     insert_before __current__ (3 lines):")
    assert all(len(line.text) <= 56 for line in rendered)
    assert not any("batch_loss_trace = []" in line.text for line in rendered)


def test_tui_log_wraps_wide_cjk_by_display_cells():
    log = TuiLog(log_level="INFO")
    log.append(
        "目前运行在 epoch 1/3，batch 9/938，当前步骤是记录训练 loss，刚完成本 batch 的 "
        "train_one_minibatch loss=2.2553。",
        kind="program",
    )

    rendered = log.render_lines(width=40, max_lines=10)

    assert len(rendered) > 1
    assert all(_display_width(line.text) <= 40 for line in rendered)


def test_tui_log_renders_user_messages_with_label_and_color_kind():
    log = TuiLog(log_level="INFO")

    log.append("学习率调整为现在的5倍吧", kind="user")

    rendered = log.render_lines(width=80, max_lines=10)

    assert rendered[0].text == "YOU       学习率调整为现在的5倍吧"
    assert rendered[0].kind == "user"


def test_tui_log_renders_execute_now_stdout_as_agent_message():
    log = TuiLog(log_level="INFO")

    log.event_sink({"event": "execute_now", "stdout": "AI 回答应该进上方 log\n"})

    rendered = log.render_lines(width=80, max_lines=10)

    assert rendered[0].text == "AGENT     AI 回答应该进上方 log"
    assert rendered[0].kind == "agent"


def test_prompt_view_uses_display_width_for_cjk_cursor_position():
    line, cursor_col = _prompt_view("学习率调整为现在的5倍吧", width=20)

    assert _display_width(line) <= 20
    assert cursor_col == _display_width(line)
    assert cursor_col <= 19
