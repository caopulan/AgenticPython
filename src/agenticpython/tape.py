from __future__ import annotations

from dataclasses import dataclass, field
import textwrap
from typing import Any


@dataclass
class Instruction:
    id: str
    source: str
    filename: str
    lineno: int
    kind: str = "instruction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "filename": self.filename,
            "lineno": self.lineno,
        }


@dataclass
class ForBlock:
    id: str
    target: str
    iter_source: str
    body: list[Instruction]
    filename: str
    lineno: int
    kind: str = "for"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "iter_source": self.iter_source,
            "filename": self.filename,
            "lineno": self.lineno,
            "body": [instruction.to_dict() for instruction in self.body],
        }


TapeItem = Instruction | ForBlock


@dataclass
class ModuleTape:
    items: list[TapeItem]
    filename: str
    _patch_counter: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "items": [item.to_dict() for item in self.items],
        }

    def make_patch_instruction(self, source: str, target: Instruction | None = None) -> Instruction:
        self._patch_counter += 1
        return Instruction(
            id=f"P{self._patch_counter:04d}",
            source=source.strip("\n"),
            filename=target.filename if target else self.filename,
            lineno=target.lineno if target else 0,
        )

    def find_instruction(self, instruction_id: str) -> tuple[list[Instruction] | list[TapeItem], int, Instruction]:
        for index, item in enumerate(self.items):
            if isinstance(item, Instruction) and item.id == instruction_id:
                return self.items, index, item
            if isinstance(item, ForBlock):
                for body_index, instruction in enumerate(item.body):
                    if instruction.id == instruction_id:
                        return item.body, body_index, instruction
        raise KeyError(f"Instruction not found: {instruction_id}")

    def to_source(self) -> str:
        lines: list[str] = []
        for item in self.items:
            if isinstance(item, Instruction):
                lines.extend(item.source.splitlines() or [""])
                continue

            lines.append(f"for {item.target} in {item.iter_source}:")
            if item.body:
                body_source = "\n".join(instruction.source for instruction in item.body)
                lines.append(textwrap.indent(body_source, "    "))
            else:
                lines.append("    pass")
        return "\n".join(lines).rstrip() + "\n"
