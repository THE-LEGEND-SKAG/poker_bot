"""Hand ranges as lists of two-card combos, built from "the top X% of hands".

Preflop, hands are ranked by the precomputed heads-up equity of their class.
Postflop, by current made-hand strength on the board (draws are not credited:
a documented simplification that makes calling ranges slightly too strong).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

from poker_ai.engine import evaluate
from poker_ai.engine.cards import Card

from .equity import Combo, all_combos
from .preflop import class_combos, classes_by_strength

TOTAL_COMBOS = 1326


@cache
def _preflop_ranked() -> tuple[Combo, ...]:
    return tuple(c for cls in classes_by_strength() for c in class_combos(cls))


def ranked_combos(board: Sequence[Card], dead: set[Card]) -> list[Combo]:
    """All combos compatible with ``dead``, strongest first."""
    if not board:
        return [c for c in _preflop_ranked() if c[0] not in dead and c[1] not in dead]
    b = list(board)
    combos = all_combos(dead)
    combos.sort(key=lambda c: evaluate([c[0], c[1]] + b), reverse=True)
    return combos


def top_fraction(ranked: Sequence[Combo], fraction: float) -> list[Combo]:
    """The strongest ``fraction`` of ``ranked`` (at least a handful of combos)."""
    fraction = min(1.0, max(0.0, fraction))
    n = max(min(len(ranked), 6), int(round(len(ranked) * fraction)))
    return list(ranked[:n])
