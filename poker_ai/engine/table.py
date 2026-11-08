"""Multi-hand session: stacks, button movement, eliminations."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .cards import Deck
from .game_state import HandState


@dataclass(frozen=True)
class TableConfig:
    num_seats: int = 2
    starting_stack: int = 10_000
    small_blind: int = 50
    big_blind: int = 100
    ante: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.num_seats <= 10:
            raise ValueError("num_seats must be 2-10")
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError("need 0 < small_blind <= big_blind")
        if self.starting_stack <= 0 or self.ante < 0:
            raise ValueError("starting_stack must be positive and ante non-negative")


class Table:
    """Runs consecutive hands. Players with no chips sit out (tournament-style
    elimination); call :meth:`rebuy` for cash-game style play."""

    def __init__(self, config: TableConfig, seed: int | None = None, stacks: Sequence[int] | None = None) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.stacks = list(stacks) if stacks is not None else [config.starting_stack] * config.num_seats
        if len(self.stacks) != config.num_seats:
            raise ValueError("stacks length must equal num_seats")
        self.button = self.rng.randrange(config.num_seats)
        self.hand_count = 0
        self.current: HandState | None = None
        self._first = True

    @property
    def active_seats(self) -> list[int]:
        return [i for i, s in enumerate(self.stacks) if s > 0]

    @property
    def is_over(self) -> bool:
        return len(self.active_seats) < 2

    def _move_button(self) -> None:
        n = self.config.num_seats
        start = self.button if self._first else self.button + 1
        for i in range(n):
            seat = (start + i) % n
            if self.stacks[seat] > 0:
                self.button = seat
                break
        self._first = False

    def start_hand(self, deck: Deck | None = None) -> HandState:
        if self.current is not None and not self.current.terminal:
            raise RuntimeError("previous hand still in progress")
        if self.is_over:
            raise RuntimeError("fewer than two players have chips")
        self._move_button()
        self.hand_count += 1
        c = self.config
        self.current = HandState(self.stacks, self.button, c.small_blind, c.big_blind, c.ante,
                                 rng=self.rng, deck=deck, hand_id=self.hand_count)
        return self.current

    def end_hand(self) -> list[int]:
        if self.current is None or not self.current.terminal:
            raise RuntimeError("no finished hand to settle")
        self.stacks = self.current.final_stacks
        return self.current.net

    def rebuy(self, seat: int, amount: int | None = None) -> None:
        self.stacks[seat] = amount if amount is not None else self.config.starting_stack
