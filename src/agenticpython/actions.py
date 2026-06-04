from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol


class ActionValidationError(ValueError):
    """Raised when model output cannot be safely applied to the tape."""


@dataclass(frozen=True)
class PatchOperation:
    op: str
    target: str | None = None
    code: str | None = None


@dataclass(frozen=True)
class AgentAction:
    operations: list[PatchOperation]
    resume: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": [
                {"op": operation.op, "target": operation.target, "code": operation.code}
                for operation in self.operations
            ],
            "resume": self.resume,
        }


class DecisionClient(Protocol):
    def decide(self, context: dict[str, Any]) -> AgentAction:
        ...


class ScriptedActionClient:
    """Deterministic action fixture for tests and local runtime validation."""

    def __init__(self, actions_by_trigger: dict[str, dict[str, Any] | str]) -> None:
        self.actions_by_trigger = actions_by_trigger
        self.calls: list[dict[str, Any]] = []

    def decide(self, context: dict[str, Any]) -> AgentAction:
        self.calls.append(context)
        raw_action = self.actions_by_trigger.get(context["trigger"], {"operations": [], "resume": True})
        if isinstance(raw_action, str):
            return parse_action_response(raw_action)
        return _action_from_object(raw_action)


def parse_action_response(response_text: str) -> AgentAction:
    parsed = _extract_json_object(response_text)
    return _action_from_object(parsed)


def _extract_json_object(response_text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(response_text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(response_text[index:])
            return parsed
        except json.JSONDecodeError:
            continue
    raise ActionValidationError("No JSON object found in action response")


def _action_from_object(raw_action: Any) -> AgentAction:
    if not isinstance(raw_action, dict):
        raise ActionValidationError("Action response must be a JSON object")

    raw_operations = raw_action.get("operations", [])
    if not isinstance(raw_operations, list):
        raise ActionValidationError("operations must be a list")

    operations = [_operation_from_object(raw_operation) for raw_operation in raw_operations]
    if raw_action.get("stop") is True:
        operations.append(PatchOperation(op="stop"))
    return AgentAction(operations=operations, resume=bool(raw_action.get("resume", True)))


def _operation_from_object(raw_operation: Any) -> PatchOperation:
    if not isinstance(raw_operation, dict):
        raise ActionValidationError("Each operation must be a JSON object")

    op = raw_operation.get("op")
    if op not in {
        "insert_before",
        "insert_after",
        "replace",
        "delete",
        "execute_now",
        "resume",
        "stop",
    }:
        raise ActionValidationError(f"Unsupported operation: {op}")

    target = raw_operation.get("target")
    code = raw_operation.get("code")
    if op in {"insert_before", "insert_after", "replace"}:
        if not isinstance(target, str) or not target:
            raise ActionValidationError(f"{op} requires a non-empty target")
        if not isinstance(code, str) or not code.strip():
            raise ActionValidationError(f"{op} requires non-empty code")
    if op == "delete" and (not isinstance(target, str) or not target):
        raise ActionValidationError("delete requires a non-empty target")
    if op == "execute_now" and (not isinstance(code, str) or not code.strip()):
        raise ActionValidationError("execute_now requires non-empty code")

    return PatchOperation(op=op, target=target, code=code)
