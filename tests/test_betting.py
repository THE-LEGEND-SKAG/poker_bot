import pytest

from poker_ai.engine import Action, ActionType, HandState, IllegalActionError, Street

from .conftest import rigged_hand


def hu(stacks=(1000, 1000), button=0):
    return rigged_hand(list(stacks), button, 5, 10, {})


def snapshot(h: HandState):
    return (repr(h), [(p.stack, p.street_commit, p.total_commit, p.folded, p.all_in, p.acted) for p in h.players],
            h.current_bet, h.min_raise_inc, len(h.events))


# ---- positions and blinds ------------------------------------------------
def test_heads_up_button_posts_small_blind_and_acts_first_preflop():
    h = hu(button=1)
    assert (h.sb_seat, h.bb_seat) == (1, 0)
    assert h.players[1].street_commit == 5 and h.players[0].street_commit == 10
    assert h.to_act == 1


def test_heads_up_big_blind_acts_first_postflop():
    h = hu(button=0)
    h.apply(Action.call())
    h.apply(Action.check())
    assert h.street is Street.FLOP and h.to_act == 1


def test_three_handed_order():
    h = rigged_hand([1000] * 3, 0, 5, 10, {})
    assert (h.sb_seat, h.bb_seat) == (1, 2)
    assert h.to_act == 0  # UTG = button when 3-handed
    h.apply(Action.call()); h.apply(Action.call()); h.apply(Action.check())
    assert h.street is Street.FLOP and h.to_act == 1  # first left of button postflop


def test_big_blind_gets_option_after_limps():
    h = rigged_hand([1000] * 4, 0, 5, 10, {})
    h.apply(Action.call()); h.apply(Action.call()); h.apply(Action.call())
    assert h.to_act == h.bb_seat and h.street is Street.PREFLOP
    la = h.legal_actions()
    assert la.can_check and la.raise_type is ActionType.RAISE and not la.can_fold


def test_ante_is_dead_money():
    h = rigged_hand([1000] * 3, 0, 5, 10, {}, ante=2)
    assert h.pot == 6 + 15
    assert all(p.total_commit - p.street_commit == 2 for p in h.players)


# ---- legality ------------------------------------------------------------
def test_min_raise_preflop_and_reraise():
    h = hu()
    la = h.legal_actions()
    assert la.min_raise_to == 20 and la.max_raise_to == 1000
    h.apply(Action.raise_to(35))  # raise by 25
    la = h.legal_actions()
    assert la.to_call == 25 and la.min_raise_to == 60  # must raise by at least 25 more


def test_bet_vs_raise_type_enforced():
    h = hu()
    h.apply(Action.call()); h.apply(Action.check())
    la = h.legal_actions()
    assert la.raise_type is ActionType.BET and la.min_raise_to == 10
    with pytest.raises(IllegalActionError):
        h.apply(Action.raise_to(20))
    h.apply(Action.bet(20))
    assert h.legal_actions().raise_type is ActionType.RAISE


@pytest.mark.parametrize("action", [
    Action.check(),                     # facing a bet
    Action.raise_to(15),                # below min raise
    Action.raise_to(1001),              # more than the stack
    Action.bet(40),                     # wrong aggressive type preflop
])
def test_illegal_actions_rejected_without_state_change(action):
    h = hu()
    before = snapshot(h)
    with pytest.raises(IllegalActionError):
        h.apply(action)
    assert snapshot(h) == before


def test_cannot_fold_when_check_is_free():
    h = hu()
    h.apply(Action.call())
    with pytest.raises(IllegalActionError):
        h.apply(Action.fold())


def test_short_all_in_below_min_raise_is_allowed():
    h = rigged_hand([1000, 25], 0, 5, 10, {})  # seat 1 is BB with 15 behind
    h.apply(Action.raise_to(40))
    la = h.legal_actions()
    assert la.seat == 1 and la.call_amount == 15 and la.raise_type is None  # can only call all-in
    h.apply(Action.call())
    assert h.terminal  # board runs out automatically
    assert sum(h.final_stacks) == 1025


def test_incomplete_all_in_raise_does_not_reopen_betting():
    # 3 players: UTG(seat 0) raises to 100, SB(1) shoves 150 total (incomplete raise of 50),
    # BB(2) folds, UTG may only call or fold.
    h = rigged_hand([1000, 150, 1000], 0, 5, 10, {})
    h.apply(Action.raise_to(100))
    la = h.legal_actions()
    assert la.seat == 1 and la.raise_type is ActionType.RAISE and la.min_raise_to == 150 == la.max_raise_to
    h.apply(Action.raise_to(150))
    h.apply(Action.fold())  # BB
    la = h.legal_actions()
    assert la.seat == 0 and la.to_call == 50 and la.raise_type is None


def test_player_yet_to_act_may_raise_after_incomplete_all_in():
    # UTG raises to 100, SB shoves to 150 (incomplete). BB has not acted yet: may re-raise.
    h = rigged_hand([1000, 150, 1000], 0, 5, 10, {})
    h.apply(Action.raise_to(100))
    h.apply(Action.raise_to(150))
    la = h.legal_actions()
    assert la.seat == 2 and la.raise_type is ActionType.RAISE and la.min_raise_to == 240  # 150 + last full raise (90)


def test_cumulative_incomplete_raises_reopen_betting():
    # BB=100. Button 2 -> SB seat 3 (260 chips), BB seat 4 (320 chips), UTG seat 0.
    # A (seat 0) opens to 200 (full raise of 100), seat 1 calls. SB shoves 260 (+60),
    # BB shoves 320 (+60): 120 over A's last action >= 100, so A may re-raise (TDA rule).
    h = rigged_hand([5000, 5000, 5000, 260, 320], 2, 50, 100, {})
    assert (h.sb_seat, h.bb_seat, h.to_act) == (3, 4, 0)
    h.apply(Action.raise_to(200))
    h.apply(Action.call())
    h.apply(Action.fold())          # button
    h.apply(Action.raise_to(260))   # SB all-in, incomplete
    h.apply(Action.raise_to(320))   # BB all-in, incomplete
    la = h.legal_actions()
    assert la.seat == 0 and la.raise_type is ActionType.RAISE and la.min_raise_to == 420


def test_single_incomplete_raise_still_blocks():
    h = rigged_hand([5000, 5000, 5000, 260, 5000], 2, 50, 100, {})
    h.apply(Action.raise_to(200))
    h.apply(Action.fold()); h.apply(Action.fold())
    h.apply(Action.raise_to(260))  # SB shoves, +60 only
    h.apply(Action.call())         # BB calls 260
    la = h.legal_actions()
    assert la.seat == 0 and la.raise_type is None and la.to_call == 60


def test_no_raise_when_everyone_else_all_in():
    h = rigged_hand([1000, 300], 0, 5, 10, {})
    h.apply(Action.raise_to(20))
    h.apply(Action.raise_to(300))  # BB all-in
    la = h.legal_actions()
    assert la.raise_type is None and la.can_call and la.call_amount == 280


def test_uncalled_bet_returned():
    h = hu()
    h.apply(Action.raise_to(500))
    h.apply(Action.fold())
    assert h.terminal and h.final_stacks == [1010, 990]
    assert any(e["type"] == "refund" and e["amount"] == 490 for e in h.events)


def test_short_big_blind_still_requires_full_call():
    h = rigged_hand([1000, 1000, 6], 0, 5, 10, {})  # BB has only 6
    assert h.players[2].all_in and h.current_bet == 10
    la = h.legal_actions()
    assert la.seat == 0 and la.to_call == 10


def test_blinds_cover_everyone_short_hand_runs_out():
    h = rigged_hand([3, 4], 0, 5, 10, {})
    assert h.terminal and sum(h.final_stacks) == 7


def test_raise_amount_must_be_int():
    h = hu()
    with pytest.raises(IllegalActionError):
        h.apply(Action(ActionType.RAISE, 20.5))  # type: ignore[arg-type]


def test_hand_over_rejects_actions():
    h = hu()
    h.apply(Action.fold())
    with pytest.raises(IllegalActionError):
        h.apply(Action.check())
