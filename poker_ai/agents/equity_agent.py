"""Level 2: Monte Carlo equity + expected-value decisions.

Every decision runs :func:`poker_ai.agents.ev.analyze` and plays the action with
the highest EV. To avoid being perfectly predictable, it picks uniformly among
actions whose EV is within a small margin of the best (``mix_margin`` × pot).
``sims`` trades precision for speed.
"""

from __future__ import annotations

import random

from poker_ai.engine import Action, PlayerView

from .base import Agent
from .ev import analyze


class EquityAgent(Agent):
    kind = "equity"
    level = 2

    def __init__(self, name: str | None = None, seed: int | None = None, sims: int = 500, mix_margin: float = 0.02) -> None:
        super().__init__(name or "Monte Carlo EV")
        self.rng = random.Random(seed)
        self.sims = sims
        self.mix_margin = mix_margin

    def act(self, view: PlayerView) -> Action:
        a = analyze(view, sims=self.sims, rng=self.rng)
        best = a.best.ev
        margin = self.mix_margin * max(view.pot, view.big_blind)
        near = [o for o in a.options if o.ev >= best - margin]
        choice = self.rng.choice(near)
        self.last_explanation = {"agent": self.name, "action": str(choice.action), "rule": f"max EV ({choice.label})",
                                 **a.as_dict()}
        return choice.action
