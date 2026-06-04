from __future__ import annotations

import json
import threading
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
        self._btw_thread = self._codex.thread_start(**kwargs)
        self._decision_lock = threading.Lock()
        self._btw_lock = threading.Lock()

    def close(self) -> None:
        self._codex_cm.__exit__(None, None, None)

    def decide(self, context: dict[str, Any]) -> AgentAction:
        prompt = _build_prompt(context)
        with self._decision_lock:
            response = self._thread.run(prompt).final_response
        try:
            return parse_action_response(response)
        except ActionValidationError:
            repair_prompt = (
                "Your previous response was not valid AgenticPython action JSON. "
                "Return only a JSON object matching the schema, with no prose.\n\n"
                f"Previous response:\n{response}"
            )
            with self._decision_lock:
                return parse_action_response(self._thread.run(repair_prompt).final_response)

    def ask_btw(self, context: dict[str, Any]) -> str:
        prompt = _build_btw_prompt(context)
        with self._btw_lock:
            return self._btw_thread.run(prompt).final_response.strip()

    def decide_native_code(self, context: dict[str, Any]) -> str:
        prompt = _build_native_code_prompt(context)
        with self._decision_lock:
            response = self._thread.run(prompt).final_response
        try:
            return parse_native_code_response(response)
        except ActionValidationError:
            repair_prompt = (
                "Your previous response was not valid AgenticPython native-code JSON. "
                "Return only a JSON object with a non-empty string field named code.\n\n"
                f"Previous response:\n{response}"
            )
            with self._decision_lock:
                return parse_native_code_response(self._thread.run(repair_prompt).final_response)


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


def _build_btw_prompt(context: dict[str, Any]) -> str:
    return (
        "You are AgenticPython's non-interrupting /btw side conversation agent.\n"
        "Answer the user's side question in natural language. Do not return JSON. "
        "Do not propose or apply tape patches. Do not ask the runtime to pause. "
        "The Python program continues running while you answer, so be concise and "
        "base your answer only on the supplied context.\n\n"
        "BTW context JSON:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2, default=repr)}"
    )


def _build_native_code_prompt(context: dict[str, Any]) -> str:
    return (
        "You are AgenticPython's CPython-frame intervention agent. Return only JSON, no markdown.\n"
        "Allowed schema:\n"
        '{ "code": "python code to execute at the next CPython trace safepoint" }\n'
        "The code will run with the current frame globals and locals. Use only variable names "
        "shown in script_symbols or recent_source_windows, or guard them with globals().get and "
        "locals().get. Do not invent names such as epochs, batch_idx, or batches_per_epoch unless "
        "they appear in the supplied context. For inspection/debug requests, print useful fallback "
        "messages instead of raising when a value is absent. Prefer existing runtime objects such "
        "as optimizer.param_groups, lr, eval_every_epochs, logging, last_batch_summary, and "
        "batch_loss_trace when they exist. Do not import unavailable packages. Do not ask "
        "questions. Include a short print(..., flush=True) so the operator can see what changed "
        "or observed immediately.\n\n"
        "Native runtime context JSON:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2, default=repr)}"
    )


def parse_native_code_response(response_text: str) -> str:
    decoder = json.JSONDecoder()
    parsed: Any | None = None
    for index, character in enumerate(response_text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(response_text[index:])
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(parsed, dict):
        raise ActionValidationError("Native code response must be a JSON object")
    code = parsed.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ActionValidationError("Native code response requires non-empty code")
    return code
