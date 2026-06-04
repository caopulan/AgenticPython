from agenticpython.triggers import TriggerRule


def test_trigger_rule_fires_once_when_condition_becomes_true():
    rule = TriggerRule(condition="step == 4", instruction="change lr")

    assert rule.evaluate({"step": 3}) is False
    assert rule.evaluate({"step": 4}) is True
    assert rule.evaluate({"step": 4}) is False
