from __future__ import annotations

import json
from typing import Any

from .actions import ActionValidationError, AgentAction, parse_action_response


class CodexSdkDecisionClient:
    def __init__(self, model: str | None = None) -> None:
        try:
            from openai_codex import Codex, Sandbox
        except ImportError as exc:
            raise RuntimeError(
                "Codex SDK is not installed. Use Python >=3.10 and install with "
                "`python -m pip install '.[codex]'` or `python -m pip install openai-codex==0.1.0b3`."
            ) from exc

        self._codex_cm = Codex()
        self._codex = self._codex_cm.__enter__()
        kwargs: dict[str, Any] = {"sandbox": Sandbox.read_only}
        if model is not None:
            kwargs["model"] = model
        self._thread = self._codex.thread_start(**kwargs)

    def close(self) -> None:
        self._codex_cm.__exit__(None, None, None)

    def decide(self, context: dict[str, Any]) -> AgentAction:
        prompt = _build_prompt(context)
        response = self._thread.run(prompt).final_response
        try:
            return parse_action_response(response)
        except ActionValidationError:
            repair_prompt = (
                "Your previous response was not valid AgenticPython action JSON. "
                "Return only a JSON object matching the schema, with no prose.\n\n"
                f"Previous response:\n{response}"
            )
            return parse_action_response(self._thread.run(repair_prompt).final_response)


def _build_prompt(context: dict[str, Any]) -> str:
    return (
        "You are the AgenticPython intervention agent. Return only valid JSON, no markdown.\n"
        "Allowed schema:\n"
        "{\n"
        '  "operations": [\n'
        '    {"op": "insert_before|insert_after|replace|delete|execute_now|resume|stop", '
        '"target": "instruction id or __current__", "code": "python code when required"}\n'
        "  ],\n"
        '  "resume": true\n'
        "}\n"
        "Use only the supplied operation names. Prefer target __current__ when fixing the "
        "currently failing instruction. Preserve prior namespace state; do not ask questions.\n\n"
        "Runtime context JSON:\n"
        f"{json.dumps(context, ensure_ascii=True, indent=2, default=repr)}"
    )
