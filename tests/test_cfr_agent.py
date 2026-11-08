import random

import pytest

from poker_ai.agents.cfr_agent import CFRAgent, TranslationError, pseudo_harmonic_prob_a, translate
from poker_ai.agents import make_agent
from poker_ai.engine import Action, HandState
from poker_ai.sim.runner import play_hand
from poker_ai.solvers.hunl import AState

from .conftest import rigged_hand


def test_pseudo_harmonic_endpoints_and_monotonic():
    assert pseudo_harmonic_prob_a(0.5, 1.0, 0.5) == pytest.approx(1.0)
    assert pseudo_harmonic_prob_a(0.5, 1.0, 1.0) == pytest.approx(0.0)
    ps = [pseudo_harmonic_prob_a(0.5, 1.0, x) for x in (0.55, 0.7, 0.9)]
    assert ps == sorted(ps, reverse=True)


def test_translation_of_a_standard_line():
    # button (seat 0, SB) opens to 3 BB, BB calls, BB checks the flop: seat 0 to act
    h = rigged_hand([10000, 10000], 0, 50, 100, {}, board="Kc 7h 2s 5d 9c")
    h.apply(Action.raise_to(300)); h.apply(Action.call()); h.apply(Action.check())
    s, me = translate(h.view(0))
    assert me == 0 and s.history_key() == "pc/c" and s.street == 1
    expected = AState().apply("p").apply("c").apply("c")
    assert s.contrib == expected.contrib


def test_odd_bet_sizes_map_to_nearest_abstract_size():
    # 2.5 BB is 0.75 pot: pseudo-harmonic mapping sends it to "pot" (P(½ pot) = 0.43)
    for real_to, token in ((220, "h"), (230, "h"), (250, "p"), (330, "p"), (600, "p")):
        h = rigged_hand([10000, 10000], 0, 50, 100, {})
        h.apply(Action.raise_to(real_to))
        s, _ = translate(h.view(1))
        assert s.history_key() == token, (real_to, s.history_key())


def test_untranslatable_line_raises():
    h = rigged_hand([10000, 10000], 0, 50, 100, {})
    for to in (300, 900, 2700, 8100):  # four raises exceed the abstraction's preflop cap
        h.apply(Action.raise_to(to))
    with pytest.raises(TranslationError):
        translate(h.view(1))


def test_heads_up_100bb_uses_the_blueprint_and_multiway_falls_back():
    a = CFRAgent(seed=1, bucket_sims=100)
    rng = random.Random(2)
    for _ in range(30):
        h = HandState([10000, 10000], rng.randrange(2), 50, 100, rng=rng)
        play_hand(h, {0: a, 1: make_agent("rule_balanced", seed=3)})
    assert a.decisions["blueprint"] > a.decisions["fallback"]
    a.decisions = {"blueprint": 0, "fallback": 0}
    h = HandState([10000] * 3, 0, 50, 100, rng=rng)
    play_hand(h, {0: a, 1: make_agent("calling_station"), 2: make_agent("calling_station")})
    if a.decisions["blueprint"] + a.decisions["fallback"]:
        assert a.decisions["blueprint"] == 0


def test_short_stacks_fall_back_with_reason():
    a = CFRAgent(seed=1)
    h = HandState([1000, 1000], 0, 50, 100, rng=random.Random(1))
    a.act(h.view(h.to_act))
    assert a.last_explanation["source"] == "fallback" and "BB" in a.last_explanation["rule"]
