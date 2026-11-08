"""Baseline: uniformly random over action categories, random legal sizing."""

from __future__ import annotations

import random

from poker_ai.engine import Action, PlayerView

from .base import Agent


class RandomAgent(Agent):
    kind = "random"
    level = 0

    def __init__(self, name: str | None = None, seed: int | None = None) -> None:
        super().__init__(name or "Random")
        self.rng = random.Random(seed)

    def act(self, view: PlayerView) -> Action:
        la = view.legal
        assert la is not None
        options: list[Action] = []
        if la.can_fold:
            options.append(Action.fold())
        options.append(la.passive())
        if la.raise_type is not None:
            options.append(Action(la.raise_type, self.rng.randint(la.min_raise_to, la.max_raise_to)))
        action = self.rng.choice(options)
        self.last_explanation = {"agent": self.name, "action": str(action), "rule": "uniform random"}
        return action


class CallingStation(Agent):
    """Never folds, never raises. Useful as an exploitable benchmark."""

    kind = "calling_station"
    level = 0

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name or "Calling Station")

    def act(self, view: PlayerView) -> Action:
        assert view.legal is not None
        return view.legal.passive()
