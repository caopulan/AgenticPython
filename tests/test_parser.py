import pytest

from agenticpython.parser import UnsupportedEditableConstruct, parse_script


def test_parser_keeps_top_level_statements_and_stepwise_for_body():
    tape = parse_script(
        "\n".join(
            [
                "x = 1",
                "for step in range(2):",
                "    x = x + step",
                "print(x)",
            ]
        ),
        filename="demo.py",
    )

    assert [item.kind for item in tape.items] == ["instruction", "for", "instruction"]
    loop = tape.items[1]
    assert loop.target == "step"
    assert loop.iter_source == "range(2)"
    assert [instruction.source for instruction in loop.body] == ["x = x + step"]


def test_parser_rejects_nested_control_flow_inside_editable_loop():
    with pytest.raises(UnsupportedEditableConstruct, match="Nested control flow"):
        parse_script(
            "\n".join(
                [
                    "for step in range(2):",
                    "    while step < 1:",
                    "        step += 1",
                ]
            ),
            filename="bad.py",
        )
