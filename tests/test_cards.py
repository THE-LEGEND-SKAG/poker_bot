import random

import pytest

from poker_ai.engine.cards import FULL_DECK, Deck, card_str, parse_card, parse_cards, rank_of, suit_of


def test_round_trip_all_cards():
    assert len(FULL_DECK) == 52
    strs = {card_str(c) for c in FULL_DECK}
    assert len(strs) == 52
    for c in FULL_DECK:
        assert parse_card(card_str(c)) == c


def test_rank_suit():
    c = parse_card("As")
    assert rank_of(c) == 12 and suit_of(c) == 3
    assert parse_card("2c") == 0
    assert parse_card("10h") == parse_card("Th")


@pytest.mark.parametrize("bad", ["", "A", "1s", "Ax", "AsK"])
def test_parse_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_card(bad)


def test_parse_cards_formats_and_duplicates():
    assert parse_cards("As Kd") == parse_cards("AsKd") == parse_cards(["As", "Kd"])
    with pytest.raises(ValueError):
        parse_cards("As As")


def test_deck_integrity_and_determinism():
    d1, d2 = Deck(random.Random(7)), Deck(random.Random(7))
    assert d1.order == d2.order
    assert sorted(d1.order) == list(range(52))
    dealt = d1.deal(52)
    assert len(set(dealt)) == 52
    with pytest.raises(RuntimeError):
        d1.deal(1)


def test_shuffle_is_roughly_uniform():
    counts = [0] * 52
    rng = random.Random(0)
    for _ in range(5200):
        counts[Deck(rng).deal(1)[0]] += 1
    assert min(counts) > 50 and max(counts) < 160


def test_from_order_stacks_top_cards():
    top = parse_cards("As Ks Qs")
    d = Deck.from_order(top)
    assert d.deal(3) == top
    assert d.remaining == 49
    with pytest.raises(ValueError):
        Deck.from_order([1, 1])
