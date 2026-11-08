import random
from fractions import Fraction

import pytest

from poker_ai.engine import Action, ActionType, HandState, IllegalActionError, Street, Table, TableConfig, evaluate

from .conftest import rigged_hand


def random_action(h: HandState, rng: random.Random) -> Action:
    la = h.legal_actions()
    choices = []
    if la.can_fold:
        choices.append(Action.fold())
    if la.can_check:
        choices += [Action.check()] * 3
    if la.can_call:
        choices += [Action.call()] * 3
    if la.raise_type is not None:
        choices.append(Action(la.raise_type, la.min_raise_to))
        choices.append(Action(la.raise_type, la.max_raise_to))
        choices.append(Action(la.raise_type, rng.randint(la.min_raise_to, la.max_raise_to)))
    return rng.choice(choices)


def random_illegal(h: HandState, rng: random.Random) -> Action:
    la = h.legal_actions()
    opts = [Action(ActionType.RAISE, la.max_raise_to + rng.randint(1, 50)), Action(ActionType.BET, la.max_raise_to + 1)]
    if not la.can_check:
        opts.append(Action.check())
    if not la.can_fold:
        opts.append(Action.fold())
    if la.raise_type is None:
        opts.append(Action(ActionType.RAISE, la.max_raise_to))
    elif la.min_raise_to < la.max_raise_to and la.min_raise_to - 1 > la.street_commit + la.to_call:
        opts.append(Action(la.raise_type, la.min_raise_to - 1))
    return rng.choice(opts)


def reference_payouts(h: HandState) -> dict[int, Fraction]:
    """Chip-by-chip pot distribution (independent of build_pots)."""
    contrib = {p.seat: p.total_commit for p in h.players if p.in_hand}
    live = [p.seat for p in h.players if p.live]
    if len(live) == 1:
        return {live[0]: Fraction(sum(contrib.values()))}
    score = {s: evaluate(h.players[s].hole + h.board) for s in live}
    out = {s: Fraction(0) for s in live}
    for level in range(1, max(contrib.values()) + 1):
        chips = sum(1 for c in contrib.values() if c >= level)
        eligible = [s for s in live if contrib[s] >= level]
        if not eligible:
            eligible = live  # unreachable after uncalled-bet refunds
        best = max(score[s] for s in eligible)
        winners = [s for s in eligible if score[s] == best]
        for w in winners:
            out[w] += Fraction(chips, len(winners))
    return out


def check_invariants(h: HandState, total: int) -> None:
    assert sum(p.stack for p in h.players) + (0 if h.terminal else h.pot) == total or h.terminal
    for p in h.players:
        assert p.stack >= 0
        assert not p.all_in or p.stack == 0 or h.terminal
    cards = h.board + [c for p in h.players for c in p.hole]
    assert len(cards) == len(set(cards))
    assert len(h.board) == {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5, Street.SHOWDOWN: 5}[h.street] or h.terminal


@pytest.mark.parametrize("seed", range(6))
def test_random_stress_chip_conservation_and_legality(seed):
    rng = random.Random(seed)
    for _ in range(400):
        n = rng.randint(2, 9)
        bb = rng.choice([2, 10, 100])
        sb = rng.choice([bb // 2, bb]) or 1
        stacks = [rng.choice([0, rng.randint(1, 3 * bb), rng.randint(bb, 200 * bb)]) for _ in range(n)]
        if sum(1 for s in stacks if s > 0) < 2:
            stacks[0] = stacks[1] = 50 * bb
        ante = rng.choice([0, 0, max(1, bb // 10)])
        button = rng.randrange(n)
        total = sum(stacks)
        h = HandState(stacks, button, sb, bb, ante, rng=rng)
        steps = 0
        while not h.terminal:
            steps += 1
            assert steps < 500, "hand did not terminate"
            if rng.random() < 0.1:
                before = (repr(h), [(p.stack, p.street_commit) for p in h.players], h.current_bet)
                with pytest.raises(IllegalActionError):
                    h.apply(random_illegal(h, rng))
                assert before == (repr(h), [(p.stack, p.street_commit) for p in h.players], h.current_bet)
            h.apply(random_action(h, rng))
            check_invariants(h, total)
        assert sum(h.final_stacks) == total
        assert all(s >= 0 for s in h.final_stacks)
        ref = reference_payouts(h)
        n_pots = max(1, len(h.pots_awarded))
        for seat in range(n):
            assert abs(h.winnings[seat] - ref.get(seat, 0)) < n_pots, (seat, h.winnings, ref)


def test_fold_preflop_ends_hand():
    h = rigged_hand([1000, 1000], 0, 5, 10, {})
    h.apply(Action.fold())
    assert h.terminal and h.final_stacks == [995, 1005] and h.net == [-5, 5]


def test_full_hand_to_showdown_best_hand_wins():
    h = rigged_hand([1000, 1000], 0, 5, 10, {0: "Ah Kh", 1: "Qc Qd"}, board="Ac 7d 2s 9h 3c")
    h.apply(Action.raise_to(30)); h.apply(Action.call())
    for _ in range(3):
        h.apply(Action.bet(50) if h.legal_actions().can_check else Action.call())
        h.apply(Action.call())
    assert h.terminal and h.street is Street.SHOWDOWN
    assert h.final_stacks == [1180, 820]
    showdown = next(e for e in h.events if e["type"] == "showdown")
    assert showdown["hands"][0]["description"] == "Pair of Aces"


def test_all_in_preflop_runs_out_board():
    h = rigged_hand([500, 500], 0, 5, 10, {0: "2c 2d", 1: "Ac Kd"}, board="2h 7s 9c Jd 4h")
    h.apply(Action.raise_to(500)); h.apply(Action.call())
    assert h.terminal and len(h.board) == 5
    assert h.final_stacks == [1000, 0]
    assert [e["street"] for e in h.events if e["type"] == "street"] == ["flop", "turn", "river"]


def test_view_hides_opponent_cards_until_showdown():
    h = rigged_hand([1000, 1000], 0, 5, 10, {0: "Ah Kh", 1: "Qc Qd"}, board="Ac 7d 2s 9h 3c")
    v = h.view(1)
    assert v.hole == tuple(h.players[1].hole)
    assert all(p.shown_cards is None for p in v.players)
    text = repr(v.history)
    assert str(h.players[0].hole[0]) not in [str(x) for e in v.history for x in e.get("cards", [])]
    assert v.legal is None and h.view(0).legal is not None
    assert "hole" not in text
    h.apply(Action.raise_to(1000)); h.apply(Action.call())
    v = h.view(1)
    assert v.players[0].shown_cards == tuple(h.players[0].hole)


def test_table_moves_button_and_eliminates():
    t = Table(TableConfig(num_seats=3, starting_stack=1000, small_blind=5, big_blind=10), seed=3)
    buttons = []
    rng = random.Random(0)
    while not t.is_over and t.hand_count < 500:
        h = t.start_hand()
        buttons.append(h.button)
        while not h.terminal:
            la = h.legal_actions()
            h.apply(la.all_in() if rng.random() < 0.1 else la.passive())
        t.end_hand()
        assert sum(t.stacks) == 3000
        assert t.stacks[h.button] >= 0
    assert len(buttons) >= 2 and buttons[1] != buttons[0]
    if t.is_over:
        assert sorted(t.stacks)[-1] == 3000


def test_table_rejects_overlapping_hands():
    t = Table(TableConfig(), seed=1)
    t.start_hand()
    with pytest.raises(RuntimeError):
        t.start_hand()


def test_reproducible_with_seed():
    def play(seed):
        t = Table(TableConfig(num_seats=4), seed=seed)
        rng = random.Random(seed)
        for _ in range(20):
            if t.is_over:
                break
            h = t.start_hand()
            while not h.terminal:
                h.apply(random_action(h, rng))
            t.end_hand()
        return t.stacks
    assert play(11) == play(11)
