import os

import pytest


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
