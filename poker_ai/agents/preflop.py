"""The 169 preflop hand classes ("AKs", "T9o", "77") and their equity table.

The table (``poker_ai/data/preflop_equity.json``) holds each class's equity
against 1..5 random hands. Rebuild it with ``python scripts/build_preflop_table.py``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from poker_ai.engine.cards import RANKS, Card

from .equity import Combo

TABLE_PATH = Path(__file__).resolve().parents[1] / "data" / "preflop_equity.json"


def hand_class(c1: Card, c2: Card) -> str:
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if (c1 & 3) == (c2 & 3) else "o")


def all_classes() -> list[str]:
    """Row-major 13x13 grid order used by the range heatmap: pairs on the
    diagonal, suited above it, offsuit below it."""
    out = []
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            if i == j:
                out.append(RANKS[i] * 2)
            elif j < i:
                out.append(RANKS[i] + RANKS[j] + "s")
            else:
                out.append(RANKS[j] + RANKS[i] + "o")
    return out


def class_combos(cls: str) -> list[Combo]:
    r1, r2 = RANKS.index(cls[0]), RANKS.index(cls[1])
    if len(cls) == 2:
        return [(r1 * 4 + a, r1 * 4 + b) for a in range(4) for b in range(a + 1, 4)]
    if cls[2] == "s":
        return [(r1 * 4 + s, r2 * 4 + s) for s in range(4)]
    return [(r1 * 4 + a, r2 * 4 + b) for a in range(4) for b in range(4) if a != b]


@cache
def preflop_table() -> dict[str, list[float]]:
    """class -> [equity vs 1 random hand, vs 2, ..., vs 5]."""
    return json.loads(TABLE_PATH.read_text())


def preflop_equity(c1: Card, c2: Card, opponents: int = 1) -> float:
    return preflop_table()[hand_class(c1, c2)][min(max(opponents, 1), 5) - 1]


@cache
def classes_by_strength() -> list[str]:
    """Hand classes, strongest first (heads-up equity vs a random hand)."""
    t = preflop_table()
    return sorted(t, key=lambda c: t[c][0], reverse=True)
