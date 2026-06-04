from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriggerRule:
    condition: str
    instruction: str
    once: bool = True
    fired: bool = False

    def evaluate(self, namespace: dict[str, object]) -> bool:
        if self.once and self.fired:
            return False
        try:
            matched = bool(eval(self.condition, namespace, namespace))
        except NameError:
            matched = False
        if matched and self.once:
            self.fired = True
        return matched
