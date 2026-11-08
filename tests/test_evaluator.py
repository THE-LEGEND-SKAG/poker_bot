import random
from collections import Counter
from itertools import combinations

import pytest

from poker_ai.engine.cards import parse_cards
from poker_ai.engine.hand_evaluator import (
    CATEGORY_NAMES, best_five, category, category_name, describe, evaluate,
)


# ---- independent reference implementation (slow, obviously-correct) -------
def ref5(cards):
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = {c & 3 for c in cards}
    cnt = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len(suits) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_hi = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_hi = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_hi = 3
    if straight_hi is not None and flush:
        return (8, straight_hi)
    shape = [g[1] for g in groups]
    ordered = [g[0] for g in groups]
    if shape == [4, 1]:
        return (7, *ordered)
    if shape == [3, 2]:
        return (6, *ordered)
    if flush:
        return (5, *ranks)
    if straight_hi is not None:
        return (4, straight_hi)
    if shape == [3, 1, 1]:
        return (3, *ordered)
    if shape == [2, 2, 1]:
        return (2, *ordered)
    if shape == [2, 1, 1, 1]:
        return (1, *ordered)
    return (0, *ranks)


def ref7(cards):
    return max(ref5(c) for c in combinations(cards, 5))


def sign(x):
    return (x > 0) - (x < 0)


# ---- known hands --------------------------------------------------------
@pytest.mark.parametrize(
    "cards,name",
    [
        ("As Ks Qs Js Ts 2d 3c", "Royal Flush"),
        ("9h 8h 7h 6h 5h Ad Ac", "Straight Flush"),
        ("Ah 2h 3h 4h 5h Kd Kc", "Straight Flush"),
        ("Qd Qc Qh Qs 2d 3c 4h", "Four of a Kind"),
        ("Kd Kc Kh 2s 2d 3c 4h", "Full House"),
        ("Kd Kc Kh 2s 2d 2c 4h", "Full House"),
        ("Ad 9d 7d 4d 2d Ks Kc", "Flush"),
        ("Ad 2c 3h 4s 5d Kc 9h", "Straight"),
        ("Td Jc Qh Ks Ad 2c 3h", "Straight"),
        ("7d 7c 7h As Kd 2c 4h", "Three of a Kind"),
        ("7d 7c 9h 9s Kd Kc 4h", "Two Pair"),
        ("7d 7c 9h Js Kd 2c 4h", "One Pair"),
        ("7d 8c 9h Js Kd 2c 4h", "High Card"),
    ],
)
def test_categories(cards, name):
    assert category_name(evaluate(parse_cards(cards))) == name


def test_ace_low_straight_is_lowest_straight():
    wheel = evaluate(parse_cards("Ad 2c 3h 4s 5d 9c Kh"))
    six_high = evaluate(parse_cards("2c 3h 4s 5d 6c 9d Kh"))
    assert category_name(wheel) == "Straight"
    assert six_high > wheel
    assert describe(wheel) == "Straight, Five high"


def test_kickers_matter_and_ties_are_exact():
    board = parse_cards("Ah Kd 7c 4s 2h")
    a = evaluate(parse_cards("Ac Qd") + board)  # pair of aces, K Q 7
    b = evaluate(parse_cards("As Jd") + board)  # pair of aces, K J 7
    assert a > b
    # board plays: both hands use the same five board cards
    board2 = parse_cards("Ah Kh Qh Jh Th")
    assert evaluate(parse_cards("2c 3d") + board2) == evaluate(parse_cards("4c 5d") + board2)


def test_two_pair_uses_best_two_pairs_and_kicker():
    # three pairs available: best two (K, 9) + best kicker (7 beats the 5 of the third pair)
    s = evaluate(parse_cards("Kd Kc 9h 9s 5d 5c 7h"))
    assert describe(s) == "Two Pair, Kings and Nines"
    assert s == evaluate(parse_cards("Kd Kc 9h 9s 7h 2c 3d"))


def test_full_house_prefers_higher_trips_as_set():
    s = evaluate(parse_cards("2d 2c 2h Ks Kd Kc 4h"))
    assert describe(s) == "Full House, Kings full of Twos"


def test_flush_beats_straight_on_same_board():
    straight = evaluate(parse_cards("9c Td Jh Qs Kd 2d 3d"))
    flush = evaluate(parse_cards("4d 7d Td Jh Qs Kd 2d"))
    assert flush > straight


def test_best_five_returns_matching_cards():
    cards = parse_cards("As Ks Qs Js Ts 2d 3c")
    score, five = best_five(cards)
    assert score == evaluate(cards) and set(five) == set(parse_cards("As Ks Qs Js Ts"))


def test_random_7card_matches_reference_ordering():
    rng = random.Random(1234)
    deck = list(range(52))
    for _ in range(3000):
        rng.shuffle(deck)
        a, b = deck[:7], deck[7:14]
        ea, eb = evaluate(a), evaluate(b)
        ra, rb = ref7(a), ref7(b)
        assert category(ea) == ra[0]
        assert sign(ea - eb) == sign((ra > rb) - (ra < rb))


def test_random_5_and_6_card_hands_match_reference():
    rng = random.Random(99)
    deck = list(range(52))
    for n in (5, 6):
        for _ in range(1500):
            rng.shuffle(deck)
            a, b = deck[:n], deck[n : 2 * n]
            ra, rb = (max(ref5(c) for c in combinations(h, 5)) for h in (a, b))
            assert sign(evaluate(a) - evaluate(b)) == sign((ra > rb) - (ra < rb))


@pytest.mark.slow
def test_exhaustive_five_card_distribution():
    """All 2,598,960 five-card hands produce the textbook category counts."""
    counts = [0] * 9
    royal = 0
    for combo in combinations(range(52), 5):
        s = evaluate(combo)
        counts[s >> 20] += 1
        if s >> 20 == 8 and (s >> 16) & 0xF == 12:
            royal += 1
    expected = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
    assert counts == expected, dict(zip(CATEGORY_NAMES, counts))
    assert royal == 4
