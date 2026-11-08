"""Hand-strength features shared by the heuristic agents.

* ``chen_score``      Bill Chen's preflop formula (≈ -1 .. 20).
* ``hand_strength``   fraction of opponent holdings we beat *right now* on this
                      board (enumerates all 1,081 or fewer combos; no future cards).
* ``count_outs``      cards that improve us to a strong made hand by the next street.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

from poker_ai.engine import evaluate
from poker_ai.engine.cards import Card
from poker_ai.engine.hand_evaluator import FLUSH, ONE_PAIR, STRAIGHT, TWO_PAIR

_CHEN_HIGH = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}


def chen_score(hole: Sequence[Card]) -> float:
    r1, r2 = sorted((hole[0] >> 2, hole[1] >> 2), reverse=True)
    suited = (hole[0] & 3) == (hole[1] & 3)
    score = _CHEN_HIGH.get(r1, (r1 + 2) / 2)
    if r1 == r2:
        return max(5.0, score * 2)
    if suited:
        score += 2
    gap = r1 - r2 - 1
    score -= (0, 1, 2, 4)[gap] if gap < 4 else 5
    if gap <= 1 and r1 < 10:  # both below queen, connected or one-gap
        score += 1
    return math.ceil(score)


def hand_strength(hole: Sequence[Card], board: Sequence[Card], opponents: int = 1) -> float:
    """P(beat a uniformly random holding), ties count half, raised to #opponents."""
    if len(board) < 3:
        raise ValueError("hand_strength needs a flop")
    dead = set(hole) | set(board)
    ours = evaluate(list(hole) + list(board))
    ahead = tied = total = 0
    rest = [c for c in range(52) if c not in dead]
    b = list(board)
    for a, c in combinations(rest, 2):
        theirs = evaluate([a, c] + b)
        if ours > theirs:
            ahead += 1
        elif ours == theirs:
            tied += 1
        total += 1
    hs = (ahead + tied / 2) / total
    return hs ** max(1, opponents)


def count_outs(hole: Sequence[Card], board: Sequence[Card]) -> int:
    """Unseen cards that turn a non-straight/flush hand into a straight or better,
    or a pair-or-worse hand using a hole card into two pair or better."""
    if len(board) >= 5:
        return 0
    cards = list(hole) + list(board)
    cur = evaluate(cards)
    cur_cat = cur >> 20
    if cur_cat >= STRAIGHT:
        return 0
    board_cat = evaluate(list(board)) >> 20 if len(board) >= 5 else -1
    outs = 0
    for c in range(52):
        if c in cards:
            continue
        new = evaluate(cards + [c])
        cat = new >> 20
        if cat >= STRAIGHT and cat != (evaluate(list(board) + [c]) >> 20 if len(board) == 4 else -1):
            outs += 1
        elif cur_cat <= ONE_PAIR and cat >= TWO_PAIR and cat > board_cat and cat < STRAIGHT:
            # only count improvements that use our hole cards (not just pairing the board)
            if new > evaluate(list(board) + [c] + [hole[0]]) or new > evaluate(list(board) + [c] + [hole[1]]):
                outs += 1
    return outs


def is_flush_or_straight_draw(hole: Sequence[Card], board: Sequence[Card]) -> bool:
    cards = list(hole) + list(board)
    suits = [0] * 4
    for c in cards:
        suits[c & 3] += 1
    if max(suits) == 4 and any(suits[h & 3] == 4 for h in hole):
        return True
    ranks = {c >> 2 for c in cards}
    if 12 in ranks:
        ranks.add(-1)
    return any(len(ranks & set(range(lo, lo + 5))) == 4 and any((h >> 2) in range(lo, lo + 5) for h in hole)
               for lo in range(-1, 9))


def made_category(hole: Sequence[Card], board: Sequence[Card]) -> int:
    return evaluate(list(hole) + list(board)) >> 20


__all__ = ["FLUSH", "chen_score", "count_outs", "hand_strength", "is_flush_or_straight_draw", "made_category"]
