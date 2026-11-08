import random

import pytest

from poker_ai.agents.equity import equity
from poker_ai.agents.ev import analyze, candidate_sizes, mdf_fold
from poker_ai.agents.preflop import all_classes, class_combos, classes_by_strength, hand_class, preflop_table
from poker_ai.agents.ranges import ranked_combos, top_fraction
from poker_ai.engine import Action, ActionType, parse_cards

from .conftest import rigged_hand

P = parse_cards


# Published heads-up equities vs one random hand (e.g. PokerStove).
@pytest.mark.parametrize("hand,expected", [("As Ah", 0.852), ("Ks Kh", 0.824), ("As Kd", 0.654), ("7c 2d", 0.346)])
def test_preflop_equity_matches_published_values(hand, expected):
    r = equity(P(hand), [], 1, 40000, random.Random(5))
    assert abs(r.equity - expected) < max(0.008, 4 * r.stderr)


def test_multiway_equity_drops():
    rng = random.Random(1)
    one = equity(P("As Ah"), [], 1, 8000, rng).equity
    three = equity(P("As Ah"), [], 3, 8000, rng).equity
    assert abs(three - 0.64) < 0.02 and three < one


def test_river_is_exact():
    r = equity(P("As Ks"), P("Qs Js Ts 2d 3c"), 1)
    assert r.exact and r.equity == 1.0 and r.samples == 990
    board_plays = equity(P("2c 3d"), P("Ah Kh Qh Jh Th"), 1)
    assert board_plays.equity == 0.5 and board_plays.tie == 1.0


def test_equity_vs_range_is_lower_than_vs_random():
    rng = random.Random(2)
    strong = top_fraction(ranked_combos([], set(P("Ks Qd"))), 0.05)
    assert equity(P("Ks Qd"), [], [strong], 6000, rng).equity < equity(P("Ks Qd"), [], 1, 6000, rng).equity - 0.15


def test_incompatible_range_rejected():
    with pytest.raises(ValueError):
        equity(P("As Ah"), P("Ad Ac 2s 3s 4s"), [[(P("As")[0], P("Kd")[0])]])


def test_preflop_classes_and_table():
    classes = all_classes()
    assert len(classes) == len(set(classes)) == 169
    assert sum(len(class_combos(c)) for c in classes) == 1326
    t = preflop_table()
    assert set(t) == set(classes)
    order = classes_by_strength()
    assert order[0] == "AA" and order[-1] in ("32o", "42o", "62o", "72o", "52o")
    assert all(row == sorted(row, reverse=True) for row in t.values())  # more opponents -> less equity
    assert hand_class(*P("Ah Kh")) == "AKs" and hand_class(*P("Kd Ah")) == "AKo" and hand_class(*P("7c 7d")) == "77"


def test_fold_model_properties():
    # a balanced defender would make a pure bluff break even; ours folds less, so bluffs lose
    for risk, reward in [(50, 100), (100, 100), (300, 100)]:
        f = mdf_fold(None, 0, risk, reward)  # type: ignore[arg-type]
        assert f < risk / (risk + reward)
        assert f * reward - (1 - f) * risk < 0
    assert mdf_fold(None, 0, 0, 100) == 0  # type: ignore[arg-type]


def test_fold_ev_is_zero_and_nuts_never_fold():
    h = rigged_hand([5000, 5000], 0, 50, 100, {0: "As Ks", 1: "Qd Qc"}, board="Qs Js Ts 2d 3c")
    for _ in range(6):
        h.apply(h.legal_actions().passive())
    h.apply(Action.bet(200))  # seat 1 (first to act postflop) bets, seat 0 holds the royal flush
    a = analyze(h.view(0), 400, random.Random(0))
    fold = next(o for o in a.options if o.action.type is ActionType.FOLD)
    assert fold.ev == 0
    assert a.equity.equity == 1.0 and a.best.action.type is ActionType.RAISE


def test_trash_folds_to_a_raise_and_aces_raise():
    h = rigged_hand([10000, 10000], 0, 50, 100, {0: "As Ah", 1: "7c 2d"})
    assert analyze(h.view(0), 800, random.Random(1)).best.action.type is ActionType.RAISE
    h.apply(Action.raise_to(300))
    assert analyze(h.view(1), 800, random.Random(1)).best.action.type is ActionType.FOLD


def test_candidate_sizes_are_legal_and_unique():
    rng = random.Random(3)
    from poker_ai.engine import HandState
    for _ in range(200):
        h = HandState([rng.randint(150, 20000) for _ in range(3)], 0, 50, 100, rng=rng)
        while not h.terminal:
            la = h.legal_actions()
            if la.raise_type is not None:
                sizes = candidate_sizes(h.view(h.to_act))
                tos = [t for _, t in sizes]
                assert len(tos) == len(set(tos)) and tos
                for t in tos:
                    assert la.is_legal(Action(la.raise_type, t))
            h.apply(la.raise_action(la.min_raise_to) if la.raise_type and rng.random() < 0.3 else la.passive())


def test_size_labels_match_amounts():
    """Regression: with a limper, the 2.5 BB open is 350 and must be labelled 3.5 BB."""
    h = rigged_hand([10000] * 3, 0, 50, 100, {})
    h.apply(Action.call())  # button limps; SB to act
    h.apply(Action.call())  # SB completes; BB faces two limpers... BB can raise
    v = h.view(h.to_act)
    for label, to in candidate_sizes(v):
        if label.endswith("BB"):
            assert float(label.split()[0]) * 100 == to, (label, to)
    rng = random.Random(4)
    from poker_ai.engine import HandState
    for _ in range(200):  # labels always describe the real amount
        h = HandState([rng.randint(300, 20000) for _ in range(4)], 0, 50, 100, rng=rng)
        while not h.terminal:
            la = h.legal_actions()
            if la.raise_type is not None and h.street.label == "preflop":
                for label, to in candidate_sizes(h.view(h.to_act)):
                    if label.endswith(" BB"):
                        assert abs(float(label.split()[0]) * 100 - to) < 1e-6
            h.apply(la.passive() if rng.random() < 0.8 or la.raise_type is None else la.raise_action(la.min_raise_to))


def test_realisation_is_bounded_and_weak_hands_realise_less():
    """Regression: aggression scaling once pushed realisation to 10%, making Level 3 fold far too much."""
    h = rigged_hand([10000, 10000], 0, 50, 100, {0: "7c 2d", 1: "As Ah"})
    strong = rigged_hand([10000, 10000], 0, 50, 100, {0: "As Ah", 1: "7c 2d"})
    for scale in (0.0, 1.0, 1.5, 5.0):
        weak_r = analyze(h.view(0), 300, random.Random(0), aggression_scale=scale).realization
        strong_r = analyze(strong.view(0), 300, random.Random(0), aggression_scale=scale).realization
        assert 0.4 <= weak_r <= strong_r <= 1.0
