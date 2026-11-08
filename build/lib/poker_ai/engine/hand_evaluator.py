"""Deterministic 5-7 card poker hand evaluator.

``evaluate(cards)`` returns an ``int`` score; a higher score is a better hand and
equal scores are exact ties. The score is::

    category << 20 | r1 << 16 | r2 << 12 | r3 << 8 | r4 << 4 | r5

where ``category`` is 0 (high card) .. 8 (straight flush) and ``r1..r5`` are the
ranks (0 = deuce .. 12 = ace) that break ties, most significant first.

The evaluator is table driven: 13-bit rank masks index precomputed tables of
straight highs and "top five ranks" kicker encodings. It is exhaustively checked
against a naive best-of-21-combinations evaluator and against the known
2,598,960-hand category distribution in ``tests/test_evaluator.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

from .cards import Card, RANKS

HIGH_CARD, ONE_PAIR, TWO_PAIR, THREE_OF_A_KIND, STRAIGHT, FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH = range(9)

CATEGORY_NAMES = (
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
)

_ACE = 12


def _build_tables() -> tuple[list[int], list[int]]:
    straight_high = [-1] * 8192
    top5 = [0] * 8192
    windows = [(0b11111 << lo, lo + 4) for lo in range(9)]  # 2-6 .. T-A
    wheel = (1 << _ACE) | 0b1111  # A-2-3-4-5, high card is the five (rank 3)
    for mask in range(8192):
        best = -1
        for w, hi in windows:
            if mask & w == w:
                best = hi
        if best < 0 and mask & wheel == wheel:
            best = 3
        straight_high[mask] = best
        # top five set bits packed into 20 bits
        code = 0
        n = 0
        for r in range(12, -1, -1):
            if mask >> r & 1:
                code |= r << (16 - 4 * n)
                n += 1
                if n == 5:
                    break
        top5[mask] = code
    return straight_high, top5


STRAIGHT_HIGH, TOP5 = _build_tables()


def evaluate(cards: Sequence[Card]) -> int:
    """Score the best five-card hand from 5, 6 or 7 cards."""
    counts = [0] * 13
    suit_counts = [0, 0, 0, 0]
    suit_masks = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        s = c & 3
        counts[r] += 1
        suit_counts[s] += 1
        suit_masks[s] |= 1 << r

    # With at most 7 cards a flush excludes quads and full houses, so a flush
    # (or straight flush) is always the best hand when present.
    for s in range(4):
        if suit_counts[s] >= 5:
            m = suit_masks[s]
            sh = STRAIGHT_HIGH[m]
            if sh >= 0:
                return (STRAIGHT_FLUSH << 20) | (sh << 16)
            return (FLUSH << 20) | TOP5[m]

    rank_mask = suit_masks[0] | suit_masks[1] | suit_masks[2] | suit_masks[3]

    quad = -1
    trips: list[int] = []
    pairs: list[int] = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            quad = r
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)

    if quad >= 0:
        kicker_mask = rank_mask & ~(1 << quad)
        return (FOUR_OF_A_KIND << 20) | (quad << 16) | ((TOP5[kicker_mask] >> 16) << 12)

    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        p = trips[1] if len(trips) > 1 else -1
        if pairs and pairs[0] > p:
            p = pairs[0]
        return (FULL_HOUSE << 20) | (t << 16) | (p << 12)

    sh = STRAIGHT_HIGH[rank_mask]
    if sh >= 0:
        return (STRAIGHT << 20) | (sh << 16)

    if trips:
        t = trips[0]
        k = TOP5[rank_mask & ~(1 << t)]
        return (THREE_OF_A_KIND << 20) | (t << 16) | ((k >> 4) & 0xFF00)

    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        k = TOP5[rank_mask & ~(1 << p1) & ~(1 << p2)] >> 16
        return (TWO_PAIR << 20) | (p1 << 16) | (p2 << 12) | (k << 8)

    if pairs:
        p = pairs[0]
        k = TOP5[rank_mask & ~(1 << p)]
        return (ONE_PAIR << 20) | (p << 16) | ((k >> 4) & 0xFFF0)

    return (HIGH_CARD << 20) | TOP5[rank_mask]


def evaluate_hand(hole: Iterable[Card], board: Iterable[Card]) -> int:
    return evaluate(list(hole) + list(board))


def category(score: int) -> int:
    return score >> 20


def category_name(score: int) -> str:
    cat = score >> 20
    if cat == STRAIGHT_FLUSH and (score >> 16) & 0xF == _ACE:
        return "Royal Flush"
    return CATEGORY_NAMES[cat]


_PLURAL = {"Six": "Sixes"}


def _rank_name(r: int) -> str:
    return ("Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King", "Ace")[r]


def _plural(r: int) -> str:
    n = _rank_name(r)
    return _PLURAL.get(n, n + "s")


def describe(score: int) -> str:
    """Human readable description, e.g. ``"Full House, Kings full of Fives"``."""
    cat = score >> 20
    r1 = (score >> 16) & 0xF
    r2 = (score >> 12) & 0xF
    if cat == STRAIGHT_FLUSH:
        return "Royal Flush" if r1 == _ACE else f"Straight Flush, {_rank_name(r1)} high"
    if cat == FOUR_OF_A_KIND:
        return f"Four of a Kind, {_plural(r1)}"
    if cat == FULL_HOUSE:
        return f"Full House, {_plural(r1)} full of {_plural(r2)}"
    if cat == FLUSH:
        return f"Flush, {_rank_name(r1)} high"
    if cat == STRAIGHT:
        return f"Straight, {_rank_name(r1)} high"
    if cat == THREE_OF_A_KIND:
        return f"Three of a Kind, {_plural(r1)}"
    if cat == TWO_PAIR:
        return f"Two Pair, {_plural(r1)} and {_plural(r2)}"
    if cat == ONE_PAIR:
        return f"Pair of {_plural(r1)}"
    return f"High Card, {_rank_name(r1)}"


def best_five(cards: Sequence[Card]) -> tuple[int, tuple[Card, ...]]:
    """Return ``(score, five_cards)`` for display (slow; not for simulations)."""
    target = evaluate(cards)
    for combo in combinations(cards, 5):
        if evaluate(combo) == target:
            return target, combo
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "CATEGORY_NAMES",
    "HIGH_CARD",
    "ONE_PAIR",
    "TWO_PAIR",
    "THREE_OF_A_KIND",
    "STRAIGHT",
    "FLUSH",
    "FULL_HOUSE",
    "FOUR_OF_A_KIND",
    "STRAIGHT_FLUSH",
    "RANKS",
    "best_five",
    "category",
    "category_name",
    "describe",
    "evaluate",
    "evaluate_hand",
]
