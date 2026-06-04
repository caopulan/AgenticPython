from __future__ import annotations

import ast
import textwrap

from .tape import ForBlock, Instruction, ModuleTape, TapeItem


class UnsupportedEditableConstruct(ValueError):
    """Raised when v1 would otherwise silently misrepresent Python semantics."""


class _IdFactory:
    def __init__(self) -> None:
        self._instruction_counter = 0
        self._loop_counter = 0

    def instruction_id(self) -> str:
        self._instruction_counter += 1
        return f"I{self._instruction_counter:04d}"

    def loop_id(self) -> str:
        self._loop_counter += 1
        return f"L{self._loop_counter:04d}"


_UNSUPPORTED_LOOP_BODY = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Return,
    ast.Break,
    ast.Continue,
)


def parse_script(source: str, filename: str = "<agentpython>") -> ModuleTape:
    tree = ast.parse(source, filename=filename)
    id_factory = _IdFactory()
    items: list[TapeItem] = []

    for statement in tree.body:
        if isinstance(statement, ast.For):
            items.append(_parse_for_block(statement, source, filename, id_factory))
        elif isinstance(statement, (ast.AsyncFor, ast.While)):
            raise UnsupportedEditableConstruct(
                f"Top-level {type(statement).__name__} is not supported in editable v1"
            )
        else:
            items.append(_instruction_from_statement(statement, source, filename, id_factory))

    return ModuleTape(items=items, filename=filename)


def _parse_for_block(
    statement: ast.For,
    source: str,
    filename: str,
    id_factory: _IdFactory,
) -> ForBlock:
    if statement.orelse:
        raise UnsupportedEditableConstruct("for/else is not supported in editable v1")

    body: list[Instruction] = []
    for body_statement in statement.body:
        if isinstance(body_statement, _UNSUPPORTED_LOOP_BODY):
            raise UnsupportedEditableConstruct(
                f"Nested control flow is not supported in editable loop body: "
                f"{type(body_statement).__name__} at line {body_statement.lineno}"
            )
        body.append(_instruction_from_statement(body_statement, source, filename, id_factory))

    return ForBlock(
        id=id_factory.loop_id(),
        target=ast.unparse(statement.target),
        iter_source=ast.unparse(statement.iter),
        body=body,
        filename=filename,
        lineno=statement.lineno,
    )


def _instruction_from_statement(
    statement: ast.stmt,
    source: str,
    filename: str,
    id_factory: _IdFactory,
) -> Instruction:
    segment = ast.get_source_segment(source, statement)
    if segment is None:
        segment = ast.unparse(statement)
    return Instruction(
        id=id_factory.instruction_id(),
        source=textwrap.dedent(segment).strip("\n"),
        filename=filename,
        lineno=getattr(statement, "lineno", 0),
    )
