import os

import pytest

from agenticpython.actions import ActionValidationError
from agenticpython.codex_client import _build_native_code_prompt, parse_native_code_response


def test_parse_native_code_response_extracts_code_json():
    code = parse_native_code_response(
        """
        {"code": "optimizer.param_groups[0]['lr'] = 0.01\\nprint('lr changed')"}
        """
    )

    assert "optimizer.param_groups" in code
    assert "lr changed" in code


def test_parse_native_code_response_rejects_empty_code():
    with pytest.raises(ActionValidationError, match="non-empty code"):
        parse_native_code_response('{"code": ""}')


def test_native_code_prompt_requires_flushed_prints():
    prompt = _build_native_code_prompt({"user_instruction": "现在多少iter了"})

    assert "print(..., flush=True)" in prompt


def test_native_code_prompt_warns_about_exec_local_function_state():
    prompt = _build_native_code_prompt({"user_instruction": "hook optimizer.step"})

    assert "functions created by injected code" in prompt
    assert "default arguments" in prompt


@pytest.mark.skipif(
    os.environ.get("AGENTICPYTHON_RUN_CODEX_SMOKE") != "1",
    reason="Codex SDK smoke test is opt-in because it calls local Codex auth.",
)
def test_codex_sdk_client_returns_structured_action():
    from agenticpython.codex_client import CodexSdkDecisionClient

    client = CodexSdkDecisionClient()
    try:
        action = client.decide(
            {
                "trigger": "human",
                "user_instruction": "Return a resume-only action.",
                "current_instruction": None,
                "recent_events": [],
                "future_tape": [],
            }
        )
    finally:
        client.close()

    assert action.resume is True
