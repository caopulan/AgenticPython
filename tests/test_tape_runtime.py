import json
import runpy

from agenticpython.actions import ScriptedActionClient
from agenticpython.runtime import AgenticRunner
from agenticpython.triggers import TriggerRule


def test_replaces_failing_instruction_and_retries_without_rerunning_prior_context(tmp_path):
    script = tmp_path / "error_demo.py"
    script.write_text(
        "\n".join(
            [
                "events = []",
                "events.append('before')",
                "a = 1 / 0",
                "events.append(f'after:{a}')",
            ]
        ),
        encoding="utf-8",
    )

    client = ScriptedActionClient(
        {
            "error": {
                "operations": [
                    {"op": "replace", "target": "__current__", "code": "a = 1 / 1"}
                ],
                "resume": True,
            }
        }
    )
    runner = AgenticRunner.from_path(
        script,
        decision_client=client,
        out_dir=tmp_path / "run",
    )

    result = runner.run()

    assert result.namespace["events"] == ["before", "after:1.0"]
    assert result.namespace["a"] == 1.0
    assert client.calls[0]["trigger"] == "error"
    journal = (tmp_path / "run" / "journal.jsonl").read_text(encoding="utf-8")
    assert '"status": "error"' in journal
    assert '"op": "replace"' in journal


def test_human_trigger_patches_future_loop_steps_and_replay_matches(tmp_path):
    script = tmp_path / "training_demo.py"
    script.write_text(
        "\n".join(
            [
                "lr = 0.1",
                "losses = []",
                "evals = []",
                "for step in range(6):",
                "    losses.append((step, lr))",
                "print(losses)",
                "print(evals)",
            ]
        ),
        encoding="utf-8",
    )

    client = ScriptedActionClient(
        {
            "human": {
                "operations": [
                    {
                        "op": "insert_before",
                        "target": "__current__",
                        "code": "if step >= 4:\n    lr = 0.01",
                    },
                    {
                        "op": "insert_after",
                        "target": "__current__",
                        "code": "if step >= 4:\n    evals.append(step)",
                    },
                ],
                "resume": True,
            }
        }
    )
    runner = AgenticRunner.from_path(
        script,
        decision_client=client,
        triggers=[
            TriggerRule(
                condition="step == 4",
                instruction=(
                    "From step 4 onward lower lr to 0.01 and evaluate every future step."
                ),
            )
        ],
        out_dir=tmp_path / "run",
    )

    result = runner.run()

    assert result.namespace["losses"] == [
        (0, 0.1),
        (1, 0.1),
        (2, 0.1),
        (3, 0.1),
        (4, 0.01),
        (5, 0.01),
    ]
    assert result.namespace["evals"] == [4, 5]
    replay_namespace = runpy.run_path(str(tmp_path / "run" / "replay.py"))
    assert replay_namespace["losses"] == result.namespace["losses"]
    assert replay_namespace["evals"] == result.namespace["evals"]


def test_writes_final_tape_snapshot_and_journal(tmp_path):
    script = tmp_path / "simple.py"
    script.write_text("x = 1\nx = x + 1\n", encoding="utf-8")
    runner = AgenticRunner.from_path(
        script,
        decision_client=ScriptedActionClient({}),
        out_dir=tmp_path / "run",
    )

    result = runner.run()

    assert result.namespace["x"] == 2
    tape = json.loads((tmp_path / "run" / "final_tape.json").read_text(encoding="utf-8"))
    assert [item["source"] for item in tape["items"]] == ["x = 1", "x = x + 1"]
    assert (tmp_path / "run" / "journal.jsonl").exists()
