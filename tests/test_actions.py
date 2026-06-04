import pytest

from agenticpython.actions import ActionValidationError, parse_action_response


def test_parse_action_response_extracts_json_and_validates_patch_operations():
    action = parse_action_response(
        """
        Here is the patch:
        {"operations":[{"op":"replace","target":"I0001","code":"a = 1"}],"resume":true}
        """
    )

    assert action.operations[0].op == "replace"
    assert action.operations[0].target == "I0001"
    assert action.resume is True


def test_parse_action_response_rejects_unknown_operations():
    with pytest.raises(ActionValidationError, match="Unsupported operation"):
        parse_action_response('{"operations":[{"op":"teleport","target":"I0001"}]}')
