"""Card representation.

A card is a plain ``int`` in ``0..51``: ``rank * 4 + suit``.

* rank: 0 = deuce ... 12 = ace
* suit: 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades

Plain ints keep hand evaluation and Monte Carlo simulation fast and make cards
trivially hashable/serialisable. Use :func:`card_str` / :func:`parse_card` at the
edges (UI, tests, logs).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

Card = int

RANKS = "23456789TJQKA"
SUITS = "cdhs"
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

FULL_DECK: tuple[Card, ...] = tuple(range(52))


def make_card(rank: int, suit: int) -> Card:
    if not (0 <= rank < 13 and 0 <= suit < 4):
        raise ValueError(f"invalid rank/suit {rank}/{suit}")
    return rank * 4 + suit


def rank_of(card: Card) -> int:
    return card >> 2


def suit_of(card: Card) -> int:
    return card & 3


def card_str(card: Card) -> str:
    if not 0 <= card < 52:
        raise ValueError(f"invalid card {card}")
    return RANKS[card >> 2] + SUITS[card & 3]


def parse_card(text: str) -> Card:
    """Parse ``"As"``, ``"td"``, ``"10h"`` into a card int."""
    t = text.strip()
    if len(t) == 3 and t[:2] == "10":
        t = "T" + t[2]
    if len(t) != 2:
        raise ValueError(f"cannot parse card {text!r}")
    r = RANKS.find(t[0].upper())
    s = SUITS.find(t[1].lower())
    if r < 0 or s < 0:
        raise ValueError(f"cannot parse card {text!r}")
    return r * 4 + s


def parse_cards(text: str | Iterable[str]) -> list[Card]:
    """Parse ``"As Kd"`` / ``"AsKd"`` / ``["As", "Kd"]``."""
    if isinstance(text, str):
        compact = text.replace(" ", "").replace(",", "")
        if len(compact) % 2:
            raise ValueError(f"cannot parse cards {text!r}")
        parts = [compact[i : i + 2] for i in range(0, len(compact), 2)]
    else:
        parts = list(text)
    cards = [parse_card(p) for p in parts]
    if len(set(cards)) != len(cards):
        raise ValueError(f"duplicate cards in {text!r}")
    return cards


def cards_str(cards: Iterable[Card]) -> str:
    return " ".join(card_str(c) for c in cards)


class Deck:
    """A shuffled 52-card deck.

    ``Deck(rng)`` shuffles with the given :class:`random.Random`; ``Deck.from_order``
    builds a deck with a fixed order (used for duplicate matches and tests).
    Cards are dealt from the front.
    """

    __slots__ = ("_cards", "_pos")

    def __init__(self, rng: random.Random | None = None) -> None:
        cards = list(FULL_DECK)
        (rng or random.Random()).shuffle(cards)
        self._cards = cards
        self._pos = 0

    @classmethod
    def from_order(cls, order: Sequence[Card]) -> Deck:
        order = list(order)
        if len(set(order)) != len(order) or any(not 0 <= c < 52 for c in order):
            raise ValueError("deck order must contain distinct cards in 0..51")
        if len(order) < 52:  # stack the given cards on top, rest in fixed order
            rest = [c for c in FULL_DECK if c not in set(order)]
            order = order + rest
        deck = cls.__new__(cls)
        deck._cards = order
        deck._pos = 0
        return deck

    def deal(self, n: int = 1) -> list[Card]:
        if self._pos + n > len(self._cards):
            raise RuntimeError("deck exhausted")
        out = self._cards[self._pos : self._pos + n]
        self._pos += n
        return out

    @property
    def remaining(self) -> int:
        return len(self._cards) - self._pos

    @property
    def order(self) -> list[Card]:
        return list(self._cards)
