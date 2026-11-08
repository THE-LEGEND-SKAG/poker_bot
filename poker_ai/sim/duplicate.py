"""Duplicate heads-up matches: each deck is played twice with seats swapped.

Card luck largely cancels within a pair, so the per-pair average has far lower
variance than single hands. Win rate is reported in big blinds per 100 hands with
a 95% confidence interval over pairs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from poker_ai.agents.base import Agent
from poker_ai.engine import Deck, HandState

from .runner import play_hand


@dataclass(frozen=True)
class MatchResult:
    bb100: float
    ci95: float
    hands: int

    def __str__(self) -> str:
        return f"{self.bb100:+.1f} ± {self.ci95:.1f} bb/100 ({self.hands} hands)"

    @property
    def significant(self) -> bool:
        return abs(self.bb100) > self.ci95


def duplicate_match(a: Agent | Callable[[], Agent], b: Agent | Callable[[], Agent], deals: int, seed: int = 0,
                    big_blind: int = 100, stack_bb: int = 100) -> MatchResult:
    """A's win rate against B."""
    a = a() if callable(a) and not isinstance(a, Agent) else a
    b = b() if callable(b) and not isinstance(b, Agent) else b
    rng = random.Random(seed)
    pairs: list[float] = []
    for i in range(deals):
        order = list(range(52))
        rng.shuffle(order)
        total = 0
        for a_seat in (0, 1):
            agents = {a_seat: a, 1 - a_seat: b}
            h = HandState([stack_bb * big_blind] * 2, 0, big_blind // 2, big_blind, deck=Deck.from_order(order), hand_id=i)
            play_hand(h, agents)
            total += h.net[a_seat]
        pairs.append(total / 2 / big_blind)
    n = len(pairs)
    mean = sum(pairs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in pairs) / max(1, n - 1))
    return MatchResult(mean * 100, 1.96 * sd / math.sqrt(n) * 100, 2 * n)
