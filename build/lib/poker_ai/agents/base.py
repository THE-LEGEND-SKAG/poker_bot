"""Agent interface. Agents see only a PlayerView and return an Action."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from poker_ai.engine import Action, PlayerView


class Agent(ABC):
    #: short machine id, e.g. "rule_tight"
    kind: str = "agent"
    #: ladder level shown in the UI (0 = baseline)
    level: int = 0
    #: "n/a" for non-learning bots, otherwise "untrained" / "partially trained" / "trained"
    training_status: str = "n/a"

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.kind
        self.last_explanation: dict[str, Any] | None = None

    @abstractmethod
    def act(self, view: PlayerView) -> Action:
        """Return an action. ``view.legal`` is guaranteed to be set."""

    def on_hand_end(self, view: PlayerView) -> None:
        """Called once per hand after it ends (showdown cards visible in ``view``)."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "level": self.level, "training_status": self.training_status}
