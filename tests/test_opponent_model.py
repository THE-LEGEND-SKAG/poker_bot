import random

import numpy as np

from poker_ai.agents import make_agent
from poker_ai.agents.exploitative import ExploitativeAgent, stat_fold_model
from poker_ai.agents.opponent_model import (
    ALL_COMBOS, CLASS_ORDER, OpponentModel, blocked_mask, board_strength, class_grid, estimate_ranges,
)
from poker_ai.engine import Action, HandState, Table, TableConfig, parse_cards
from poker_ai.sim.runner import play_hand

from .conftest import rigged_hand


def grid(weights, view):
    return dict(zip(CLASS_ORDER, class_grid(weights, blocked_mask(list(view.hole) + list(view.board)))))


def observe_many(opponent_kind: str, hands: int = 150, seed: int = 0) -> OpponentModel:
    """Watch an opponent from seat 0 (an aggressive bot, so it faces bets) and return the learned model."""
    model = OpponentModel(me=0)
    t = Table(TableConfig(num_seats=2), seed=seed)
    agents = {0: make_agent("rule_lag", seed=seed + 100), 1: make_agent(opponent_kind, seed=seed)}
    for _ in range(hands):
        for s in (0, 1):
            t.rebuy(s)
        h = t.start_hand()
        play_hand(h, agents)
        t.end_hand()
        model.observe_hand(h.view(0))
    return model


def test_combo_indexing():
    assert len(ALL_COMBOS) == 1326 and len(set(ALL_COMBOS)) == 1326
    assert blocked_mask(parse_cards("As")).sum() == 51


def test_board_strength_percentiles():
    s, draws = board_strength(tuple(parse_cards("Kc 7h 2h")))
    i_set = ALL_COMBOS.index(tuple(sorted(parse_cards("Kd Ks"))))
    i_air = ALL_COMBOS.index(tuple(sorted(parse_cards("4c 3d"))))
    i_fd = ALL_COMBOS.index(tuple(sorted(parse_cards("Ah 9h"))))
    assert s[i_set] > 0.99 and s[i_air] < 0.2 and draws[i_fd]


def test_preflop_raise_narrows_range_to_strong_hands():
    h = rigged_hand([10000] * 3, 0, 50, 100, {})
    h.apply(Action.raise_to(300))
    g = grid(estimate_ranges(h.view(2), OpponentModel())[0], h.view(2))
    assert g["AA"] > 0.9 and g["KQs"] > 0.5 and g["72o"] < 0.1 and g["T4o"] < 0.1


def test_postflop_bet_vs_check_shift_range():
    def after(action):
        h = rigged_hand([10000, 10000], 0, 50, 100, {0: "9s 9d"}, board="Kc 7h 2h 5s 8d")
        h.apply(Action.call()); h.apply(Action.check()); h.apply(action)
        return grid(estimate_ranges(h.view(0), OpponentModel())[1], h.view(0))
    bet, check = after(Action.bet(100)), after(Action.check())
    assert bet["KQo"] > 0.8 and check["KQo"] < 0.4
    assert bet["43o"] < 0.2 and check["43o"] > 0.8


def test_only_live_opponents_get_ranges_and_my_cards_are_blocked():
    h = rigged_hand([10000] * 3, 0, 50, 100, {2: "As Ad"})
    h.apply(Action.fold())
    w = estimate_ranges(h.view(2), OpponentModel())
    assert set(w) == {1}
    aa = [i for i, c in enumerate(ALL_COMBOS) if set(c) & set(parse_cards("As Ad"))]
    assert np.all(w[1][aa] == 0)


def test_stats_distinguish_tight_from_loose():
    tight = observe_many("rule_tight").get(1)
    station = observe_many("calling_station").get(1)
    lag = observe_many("rule_lag").get(1)
    assert station["vpip"] > 0.9 and station["pfr"] < 0.05 and station["fold_to_bet"] < 0.05
    assert tight["vpip"] < lag["vpip"]
    assert lag["pfr"] > tight["pfr"]
    assert tight.hands >= 100


def test_fold_model_follows_observed_fold_rate():
    station = observe_many("calling_station", 80)
    h = rigged_hand([10000, 10000], 0, 50, 100, {})
    h.apply(Action.call()); h.apply(Action.check())
    v = h.view(1)
    default = stat_fold_model(OpponentModel())(v, 0, 100, 200)
    vs_station = stat_fold_model(station)(v, 1, 100, 200)
    assert vs_station < default / 3


def test_model_uses_only_public_information():
    """Two hands identical except for the opponent's hidden cards give identical estimates."""
    def est(opp):
        h = rigged_hand([10000, 10000], 0, 50, 100, {0: "Qs Qd", 1: opp}, board="Kc 7h 2h 5s 8d")
        h.apply(Action.raise_to(300)); h.apply(Action.call()); h.apply(Action.bet(200))
        return estimate_ranges(h.view(0), OpponentModel())[1]
    assert np.array_equal(est("As Ah"), est("4c 3d"))


def test_exploit_agent_learns_and_plays_legally():
    a = ExploitativeAgent(seed=1, sims=200)
    t = Table(TableConfig(num_seats=3), seed=4)
    agents = {0: a, 1: make_agent("calling_station"), 2: make_agent("rule_tight", seed=2)}
    for _ in range(30):
        for s in range(3):
            t.rebuy(s)
        h = t.start_hand()
        play_hand(h, agents)
        t.end_hand()
    # checking the big blind is not VPIP, so even a station sits well below 100% early on
    assert a.model.get(1).hands > 20 and a.model.get(1)["vpip"] > 0.6 > a.model.get(2)["vpip"]
    assert 0 not in a.model.seats  # never models itself


def test_exploit_agent_value_bets_station_more_than_folder():
    """With a big made hand it should bet into a station; the decision must stay legal."""
    station = observe_many("calling_station", 80)
    agent = ExploitativeAgent(seed=3, sims=300)
    agent.model = station
    agent.model.me = None
    h = rigged_hand([10000, 10000], 0, 50, 100, {0: "Ks Kd", 1: "4c 3d"}, board="Kc 7h 2h 5s 8d")
    h.apply(Action.call()); h.apply(Action.check()); h.apply(Action.check())
    action = agent.act(h.view(0))
    assert h.legal_actions().is_legal(action) and action.type.value == "bet"


def test_random_situations_never_crash():
    rng = random.Random(9)
    a = ExploitativeAgent(seed=9, sims=100)
    for _ in range(40):
        h = HandState([rng.randint(200, 20000) for _ in range(4)], rng.randrange(4), 50, 100, rng=rng)
        others = {s: make_agent("random", seed=s) for s in range(1, 4)}
        play_hand(h, {0: a, **others})
