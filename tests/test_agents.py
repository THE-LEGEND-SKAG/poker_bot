import random

import pytest

from poker_ai.agents import available_agents, make_agent
from poker_ai.agents.features import chen_score, count_outs, hand_strength
from poker_ai.engine import HandState, IllegalActionError, TableConfig, parse_cards
from poker_ai.sim.runner import play_hand, play_session

KINDS = [a["kind"] for a in available_agents()]


@pytest.mark.parametrize("kind", KINDS)
def test_agents_always_return_legal_actions_in_random_situations(kind):
    rng = random.Random(hash(kind) & 0xFFFF)
    for i in range(150):
        n = rng.randint(2, 6)
        bb = rng.choice([2, 10, 100])
        stacks = [rng.choice([rng.randint(1, 5 * bb), rng.randint(10 * bb, 300 * bb)]) for _ in range(n)]
        h = HandState(stacks, rng.randrange(n), max(1, bb // 2), bb, rng=rng)
        agents = {s: make_agent(kind, seed=i * 10 + s) for s in range(n)}
        while not h.terminal:
            v = h.view(h.to_act)
            action = agents[h.to_act].act(v)
            assert v.legal.is_legal(action), (kind, action, v.legal)
            h.apply(action)
        assert sum(h.final_stacks) == sum(stacks)


def test_views_given_to_agents_contain_no_opponent_cards():
    seen = []

    class Spy(make_agent("rule_balanced").__class__):
        def act(self, view):
            seen.append(view)
            return super().act(view)

    h = HandState([1000, 1000, 1000], 0, 5, 10, rng=random.Random(4))
    agents = {s: Spy("balanced", seed=s) for s in range(3)}
    play_hand(h, agents)
    for v in seen:
        assert all(p.shown_cards is None for p in v.players)
        # the only cards anywhere in the view are its own hole cards and the board
        cards_in_history = {c for e in v.history for c in e.get("cards", [])} | {c for e in v.history for c in e.get("board", [])}
        assert cards_in_history <= set(v.board)
        assert all(e["type"] != "showdown" for e in v.history)


def test_engine_rejects_a_cheating_agent():
    class Cheater(make_agent("random").__class__):
        def act(self, view):
            from poker_ai.engine import Action
            return Action.raise_to(view.me.stack + view.me.street_commit + 1000)

    h = HandState([1000, 1000], 0, 5, 10, rng=random.Random(1))
    with pytest.raises(IllegalActionError):
        play_hand(h, {0: Cheater(), 1: Cheater()})


def test_chen_ordering():
    order = ["As Ah", "Ks Kh", "As Ks", "As Kd", "Js Ts", "7c 2d"]
    scores = [chen_score(parse_cards(h)) for h in order]
    assert scores == sorted(scores, reverse=True)


def test_hand_strength_extremes():
    assert hand_strength(parse_cards("As Ks"), parse_cards("Qs Js Ts")) == 1.0  # royal flush
    assert hand_strength(parse_cards("2c 3d"), parse_cards("As Kh 9d")) < 0.1
    heads_up = hand_strength(parse_cards("Ac Kd"), parse_cards("Ah 7c 2d"))
    assert hand_strength(parse_cards("Ac Kd"), parse_cards("Ah 7c 2d"), opponents=3) < heads_up


def test_outs():
    assert count_outs(parse_cards("As Ks"), parse_cards("2s 7s Jd")) >= 9
    assert count_outs(parse_cards("9c 8d"), parse_cards("7s 6h 2d")) >= 8


def test_rule_bots_beat_random_over_a_session():
    cfg = TableConfig(num_seats=2, starting_stack=10_000, small_blind=50, big_blind=100)
    res = play_session([make_agent("rule_balanced", seed=1), make_agent("random", seed=2)], cfg, hands=400, seed=3)
    assert sum(r[0] for r in res) > 0


def test_tight_folds_more_than_lag():
    cfg = TableConfig(num_seats=2, starting_stack=10_000, small_blind=50, big_blind=100)

    def vpip(kind):
        played = 0
        table_agents = [make_agent(kind, seed=5), make_agent("calling_station")]
        from poker_ai.engine import Table
        t = Table(cfg, seed=9)
        for _ in range(300):
            for s in range(2):
                t.rebuy(s)
            h = t.start_hand()
            play_hand(h, dict(enumerate(table_agents)))
            t.end_hand()
            if any(e["type"] == "action" and e["seat"] == 0 and e["street"] == "preflop" and e["action"] in ("call", "raise")
                   for e in h.events):
                played += 1
        return played / 300

    assert vpip("rule_tight") < vpip("rule_balanced") < vpip("rule_lag")
