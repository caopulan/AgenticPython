from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .actions import AgentAction
from .codex_client import CodexSdkDecisionClient
from .runtime import AgenticRunner
from .triggers import TriggerRule


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentpython")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a script through the editable tape runtime")
    run_parser.add_argument("script")
    run_parser.add_argument("--triggers")
    run_parser.add_argument("--out-dir")
    run_parser.add_argument("--model")

    smoke_parser = subparsers.add_parser("llm-smoke", help="Verify Codex SDK local-auth action output")
    smoke_parser.add_argument("--model")

    args = parser.parse_args()
    if args.command == "run":
        _run(args)
    elif args.command == "llm-smoke":
        _llm_smoke(args)


def _run(args: argparse.Namespace) -> None:
    script_path = Path(args.script)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(script_path)
    client = CodexSdkDecisionClient(model=args.model)
    try:
        runner = AgenticRunner.from_path(
            script_path,
            decision_client=client,
            triggers=_load_triggers(args.triggers),
            out_dir=out_dir,
        )
        result = runner.run()
    finally:
        client.close()
    print(f"AgenticPython artifacts: {result.out_dir}")


def _llm_smoke(args: argparse.Namespace) -> None:
    client = CodexSdkDecisionClient(model=args.model)
    try:
        action = client.decide(
            {
                "trigger": "human",
                "user_instruction": "Return exactly one resume operation and resume=true.",
                "current_instruction": None,
                "error": None,
                "namespace": {},
                "recent_events": [],
                "future_tape": [],
            }
        )
    finally:
        client.close()
    print(json.dumps(_action_to_json(action), indent=2))


def _load_triggers(trigger_path: str | None) -> list[TriggerRule]:
    if trigger_path is None:
        return []
    path = Path(trigger_path)
    if path.suffix == ".jsonl":
        raw_rules = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_rules = raw if isinstance(raw, list) else [raw]
    return [
        TriggerRule(
            condition=rule["condition"],
            instruction=rule["instruction"],
            once=bool(rule.get("once", True)),
        )
        for rule in raw_rules
    ]


def _default_out_dir(script_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(".agentpython-runs") / f"{script_path.stem}-{timestamp}"


def _action_to_json(action: AgentAction) -> dict[str, Any]:
    return {
        "operations": [
            {
                key: value
                for key, value in {
                    "op": operation.op,
                    "target": operation.target,
                    "code": operation.code,
                }.items()
                if value is not None
            }
            for operation in action.operations
        ],
        "resume": action.resume,
    }


if __name__ == "__main__":
    main()
