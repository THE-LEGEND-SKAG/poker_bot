"""Hand equity estimation.

``equity(hole, board, opponents)`` estimates our share of the pot at showdown
against ``opponents`` holdings, each drawn uniformly at random or from a range
(a list of two-card combos). Ties are shared: in a k-way tie we get 1/k.

* River, one opponent: exact enumeration over the opponent's possible holdings.
* Otherwise: Monte Carlo — sample opponent holdings and the remaining board.

The standard error is reported so callers (and the Coach panel) can show how
precise an estimate is.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from poker_ai.engine import evaluate
from poker_ai.engine.cards import Card

Combo = tuple[Card, Card]
Range = Sequence[Combo] | None  # None = any two cards


@dataclass(frozen=True, slots=True)
class EquityResult:
    equity: float  # expected pot share, ties split
    win: float  # P(we win outright)
    tie: float  # P(we split)
    samples: int
    exact: bool
    stderr: float

    def as_dict(self) -> dict[str, float | int | bool]:
        return {"equity": round(self.equity, 4), "win": round(self.win, 4), "tie": round(self.tie, 4),
                "samples": self.samples, "exact": self.exact, "stderr": round(self.stderr, 4)}


def all_combos(dead: set[Card]) -> list[Combo]:
    live = [c for c in range(52) if c not in dead]
    return list(combinations(live, 2))


def _exact_river(hole: Sequence[Card], board: Sequence[Card], rng_combos: Sequence[Combo]) -> EquityResult:
    b = list(board)
    ours = evaluate(list(hole) + b)
    dead = set(hole) | set(board)
    win = tie = n = 0
    for a, c in rng_combos:
        if a in dead or c in dead:
            continue
        theirs = evaluate([a, c] + b)
        n += 1
        if ours > theirs:
            win += 1
        elif ours == theirs:
            tie += 1
    if n == 0:
        raise ValueError("opponent range has no holdings compatible with the visible cards")
    eq = (win + tie / 2) / n
    return EquityResult(eq, win / n, tie / n, n, True, 0.0)


def equity(
    hole: Sequence[Card],
    board: Sequence[Card],
    opponents: int | Sequence[Range] = 1,
    sims: int = 1000,
    rng: random.Random | None = None,
) -> EquityResult:
    """Estimate equity against ``opponents`` (a count of random hands, or one range per opponent)."""
    if len(hole) != 2 or len(board) > 5 or len(board) in (1, 2):
        raise ValueError("need 2 hole cards and a board of 0, 3, 4 or 5 cards")
    ranges: list[Range] = [None] * opponents if isinstance(opponents, int) else list(opponents)
    if not ranges:
        return EquityResult(1.0, 1.0, 0.0, 0, True, 0.0)
    rng = rng or random.Random()
    dead = set(hole) | set(board)

    if len(board) == 5 and len(ranges) == 1:
        return _exact_river(hole, board, ranges[0] if ranges[0] is not None else all_combos(dead))

    deck = [c for c in range(52) if c not in dead]
    need_board = 5 - len(board)
    my = list(hole)
    b0 = list(board)
    total = 0.0
    total_sq = 0.0
    wins = ties = 0
    done = 0
    for _ in range(sims):
        used = set()
        opp_hands: list[list[Card]] = []
        ok = True
        for r in ranges:
            if r is None:
                continue
            for _attempt in range(30):  # rejection-sample a combo not blocked by cards already used
                a, c = r[rng.randrange(len(r))]
                if a not in dead and c not in dead and a not in used and c not in used:
                    break
            else:
                ok = False
                break
            used.add(a)
            used.add(c)
            opp_hands.append([a, c])
        if not ok:
            continue
        rest = [c for c in deck if c not in used] if used else deck
        n_random = sum(1 for r in ranges if r is None)
        drawn = rng.sample(rest, need_board + 2 * n_random)
        full_board = b0 + drawn[:need_board]
        for i in range(n_random):
            opp_hands.append(drawn[need_board + 2 * i : need_board + 2 * i + 2])
        ours = evaluate(my + full_board)
        best_other = max(evaluate(h + full_board) for h in opp_hands)
        if ours > best_other:
            share = 1.0
            wins += 1
        elif ours == best_other:
            k = 1 + sum(1 for h in opp_hands if evaluate(h + full_board) == ours)
            share = 1.0 / k
            ties += 1
        else:
            share = 0.0
        total += share
        total_sq += share * share
        done += 1
    if done == 0:
        raise ValueError("opponent ranges are incompatible with the visible cards")
    mean = total / done
    var = max(0.0, total_sq / done - mean * mean)
    return EquityResult(mean, wins / done, ties / done, done, False, math.sqrt(var / done))
