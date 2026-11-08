from poker_ai.engine import Action, build_pots, split_pot
from poker_ai.engine.pots import Pot

from .conftest import rigged_hand


def test_build_pots_three_way_all_in():
    pots = build_pots({0: 100, 1: 300, 2: 300}, live={0, 1, 2})
    assert pots == [Pot(300, (0, 1, 2)), Pot(400, (1, 2))]


def test_build_pots_folded_money_is_dead_but_counted():
    pots = build_pots({0: 50, 1: 200, 2: 200, 3: 120}, live={1, 2, 3})
    assert sum(p.amount for p in pots) == 570
    assert pots[0] == Pot(50 * 4 + 70 * 3, (1, 2, 3))
    assert pots[1] == Pot(80 * 2, (1, 2))


def test_split_pot_odd_chips_left_of_button():
    assert split_pot(101, [2, 0], seat_order=[1, 2, 0]) == {2: 51, 0: 50}
    assert split_pot(100, [4], seat_order=[0, 1, 2, 3, 4]) == {4: 100}
    assert split_pot(5, [0, 1, 2], seat_order=[2, 0, 1]) == {0: 2, 1: 1, 2: 2}


def test_short_stack_wins_main_pot_big_stack_wins_side_pot():
    # seat 0 shortest with best hand; seat 1 second best; seat 2 worst.
    h = rigged_hand([100, 500, 1000], 2, 5, 10,
                    {0: "As Ah", 1: "Ks Kh", 2: "2c 7d"}, board="3s 8d 9c Jh 4c")
    # button 2 -> SB 0, BB 1, first to act = seat 2
    h.apply(Action.raise_to(1000))   # seat 2 shoves
    h.apply(Action.call())           # seat 0 calls all-in (100)
    h.apply(Action.call())           # seat 1 calls all-in (500)
    assert h.terminal
    # main pot 300 -> seat 0; side pot (400 from 1 & 2) -> seat 1; 500 uncalled back to seat 2
    assert h.final_stacks == [300, 800, 500]


def test_split_pot_on_board_straight():
    h = rigged_hand([1000, 1000], 0, 5, 10,
                    {0: "2c 3d", 1: "2h 3s"}, board="Th Js Qd Kc Ah")
    h.apply(Action.raise_to(1000)); h.apply(Action.call())
    assert h.final_stacks == [1000, 1000]


def test_odd_chip_split_three_way():
    # 3 players each put 7 chips in after an odd-sized pot: blinds 3/7
    h = rigged_hand([7, 7, 7], 0, 3, 7, {0: "2c 3d", 1: "2h 3s", 2: "2d 3c"}, board="Th Js Qd Kc Ah")
    # all are all-in from the blinds or forced to call 7
    while not h.terminal:
        h.apply(Action.call())
    assert sorted(h.final_stacks) == [7, 7, 7]


def test_odd_chip_goes_left_of_button():
    # heads-up, 2 winners split a pot with an odd chip: pot 3+... construct via ante
    h = rigged_hand([11, 10], 0, 1, 2, {0: "2c 3d", 1: "2h 3s"}, board="Th Js Qd Kc Ah", ante=1)
    # seat 0 is button/SB. Put 11 vs 10 all-in: contributions 10/10 after refund -> even split.
    h.apply(Action.raise_to(10)); h.apply(Action.call())
    assert h.final_stacks == [11, 10]


def test_folded_player_dead_money_goes_to_side_pot_winner():
    h = rigged_hand([1000, 200, 1000], 0, 5, 10,
                    {0: "Kd Kc", 1: "Qd Qc", 2: "As Ac"}, board="2s 7h 9d Tc 3h")
    # button 0 -> SB 1 (200 stack), BB 2, seat 0 acts first
    h.apply(Action.raise_to(400))   # seat 0
    h.apply(Action.call())          # seat 1 calls all-in for 200 (short)
    h.apply(Action.fold())          # seat 2 (best hand) folds 10
    # seat 0 wins: seat1's 200 main pot (vs KK > QQ) + dead 10 from BB; 200 uncalled back
    assert h.terminal
    assert h.final_stacks == [1210, 0, 990]


def test_showdown_tie_between_two_with_third_losing_side_pot():
    h = rigged_hand([100, 300, 300], 2, 5, 10,
                    {0: "9c 9d", 1: "Ah Kh", 2: "Ad Kd"}, board="As Kc 2h 7s 4c")
    h.apply(Action.raise_to(300)); h.apply(Action.call()); h.apply(Action.call())
    # main 300 -> AK and AK split (150 each); side 400 split between seats 1 and 2
    assert h.final_stacks == [0, 350, 350]
